use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

/// Error type for tokenizer operations.
#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidData(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

pub struct TamilTokenizer {
    pub vocab: HashMap<String, u32>,
    pub byte_encoder: HashMap<u8, char>,
    pub byte_decoder: HashMap<char, u8>,
}

impl Default for TamilTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TamilTokenizer {
    pub fn new() -> Self {
        let mut vocab_tuples: Vec<(String, u32)> = Vec::new();
        let (byte_encoder, byte_decoder) = Self::create_byte_mappings();

        // Step 1: Every one of the 256 bytes gets a single-char token.
        // This is what makes the tokenizer lossless: when no learned merge
        // matches, the encoder falls back to the token for the raw byte, so
        // there is always *some* id to emit and nothing is ever dropped.
        for b in 0..=255u8 {
            let mapped_char = byte_encoder[&b];
            vocab_tuples.push((mapped_char.to_string(), vocab_tuples.len() as u32));
        }

        // Step 2: Add 12 Tamil independent vowels (in codepoint order)
        let vowel_codepoints = [
            0x0B85, 0x0B86, 0x0B87, 0x0B88, 0x0B89, 0x0B8A, 0x0B8E, 0x0B8F, 0x0B90, 0x0B92, 0x0B93,
            0x0B94,
        ];
        for &cp in &vowel_codepoints {
            if let Some(ch) = std::char::from_u32(cp) {
                vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Step 3: Add 18 consonants (in codepoint order)
        let consonant_codepoints = [
            0x0B95, 0x0B99, 0x0B9A, 0x0B9E, 0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BAF,
            0x0BB0, 0x0BB2, 0x0BB5, 0x0BB4, 0x0BB3, 0x0BB1, 0x0BA9,
        ];
        for &cp in &consonant_codepoints {
            if let Some(ch) = std::char::from_u32(cp) {
                vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Step 6: Add 12 vowel signs (in codepoint order)
        let vowel_sign_codepoints = [
            0x0BBE, 0x0BBF, 0x0BC0, 0x0BC1, 0x0BC2, 0x0BC6, 0x0BC7, 0x0BC8, 0x0BCA, 0x0BCB, 0x0BCC,
            0x0BCD,
        ];

        // Step 4: Generate 18 × 12 = 216 consonant+vowel combinations (in codepoint order)
        for &consonant_cp in &consonant_codepoints {
            for &sign_cp in &vowel_sign_codepoints {
                if let (Some(c), Some(sign)) = (
                    std::char::from_u32(consonant_cp),
                    std::char::from_u32(sign_cp),
                ) {
                    vocab_tuples.push((format!("{}{}", c, sign), vocab_tuples.len() as u32));
                }
            }
        }

        // Step 5: Add 1 special character ஃ (0x0B83)
        if let Some(ch) = std::char::from_u32(0x0B83) {
            vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
        }

        // Step 6: Add 10 Tamil digits (in codepoint order)
        for cp in 0x0BE6..=0x0BEF {
            if let Some(ch) = std::char::from_u32(cp) {
                vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Step 7: Add 11 extra Tamil chars (in codepoint order)
        for cp in 0x0BF0..=0x0BFA {
            if let Some(ch) = std::char::from_u32(cp) {
                vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Convert to HashMap
        let vocab: HashMap<String, u32> = vocab_tuples.into_iter().collect();

        Self {
            vocab,
            byte_encoder,
            byte_decoder,
        }
    }

    pub fn get_sorted_vocab(&self) -> HashMap<String, u32> {
        let mut vocab_vec: Vec<_> = self.vocab.iter().map(|(k, &v)| (k.clone(), v)).collect();
        vocab_vec.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by token string lexicographically

        // Reassign IDs in sorted order (0, 1, 2, ...)
        let sorted_vocab: HashMap<String, u32> = vocab_vec
            .into_iter()
            .enumerate()
            .map(|(idx, (token, _))| (token, idx as u32))
            .collect();

        sorted_vocab
    }

    /// Create GPT-2 style byte-to-unicode mapping
    /// Maps control chars and space to printable Unicode characters for visibility
    pub fn create_byte_mappings() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut byte_encoder = HashMap::new();
        let mut byte_decoder = HashMap::new();

        // Printable ASCII chars map to themselves
        for b in 33..=126u8 {
            let ch = b as char;
            byte_encoder.insert(b, ch);
            byte_decoder.insert(ch, b);
        }

        // Control chars and space map to high Unicode range (U+0100+).
        // The loop above only touched 33..=126, so no key here can already exist.
        let mut n = 0u32;
        for b in 0..=32u8 {
            let mapped_char = std::char::from_u32(256 + n).unwrap();
            byte_encoder.insert(b, mapped_char);
            byte_decoder.insert(mapped_char, b);
            n += 1;
        }

        // Map remaining bytes 127-255 to high Unicode
        for b in 127..=255u8 {
            let mapped_char = std::char::from_u32(256 + n).unwrap();
            byte_encoder.insert(b, mapped_char);
            byte_decoder.insert(mapped_char, b);
            n += 1;
        }

        (byte_encoder, byte_decoder)
    }

    /// Generate 247 Tamil character clusters using Unicode codepoints
    #[allow(dead_code)]
    fn generate_tamil_clusters() -> Vec<String> {
        let mut clusters = Vec::new();

        // 12 independent vowels
        let vowel_codepoints = [
            0x0B85, 0x0B86, 0x0B87, 0x0B88, 0x0B89, 0x0B8A, 0x0B8E, 0x0B8F, 0x0B90, 0x0B92, 0x0B93,
            0x0B94,
        ];
        for &cp in &vowel_codepoints {
            if let Some(ch) = std::char::from_u32(cp) {
                clusters.push(ch.to_string());
            }
        }

        // 18 consonants
        let consonant_codepoints = [
            0x0B95, 0x0B99, 0x0B9A, 0x0B9E, 0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BAF,
            0x0BB0, 0x0BB2, 0x0BB5, 0x0BB4, 0x0BB3, 0x0BB1, 0x0BA9,
        ];
        for &cp in &consonant_codepoints {
            if let Some(ch) = std::char::from_u32(cp) {
                clusters.push(ch.to_string());
            }
        }

        // 12 vowel signs
        let vowel_sign_codepoints = [
            0x0BBE, 0x0BBF, 0x0BC0, 0x0BC1, 0x0BC2, 0x0BC6, 0x0BC7, 0x0BC8, 0x0BCA, 0x0BCB, 0x0BCC,
            0x0BCD,
        ];

        // Generate 18 × 12 = 216 combinations
        for &consonant_cp in &consonant_codepoints {
            for &sign_cp in &vowel_sign_codepoints {
                if let (Some(c), Some(sign)) = (
                    std::char::from_u32(consonant_cp),
                    std::char::from_u32(sign_cp),
                ) {
                    clusters.push(format!("{}{}", c, sign));
                }
            }
        }

        // 1 special character ஃ
        if let Some(ch) = std::char::from_u32(0x0B83) {
            clusters.push(ch.to_string());
        }

        clusters
    }

    /// Decode a token back to its original byte representation
    pub fn decode_token(&self, token: &str) -> Vec<u8> {
        token
            .chars()
            .filter_map(|ch| self.byte_decoder.get(&ch).copied())
            .collect()
    }

    // pub fn save_vocab(&self, path: &str) -> std::io::Result<()> {
    //     let json = serde_json::to_string_pretty(&self.vocab)
    //         .expect("Serialization failed");
    //     let mut file = File::create(path)?;
    //     file.write_all(json.as_bytes())?;
    //     Ok(())
    // }

    pub fn save_vocab(&self, path: &str) -> Result<()> {
        // use std::collections::BTreeMap;

        // Build (id -> token) so we can serialize in strict ID order
        let mut by_id: Vec<(u32, &str)> = self
            .vocab
            .iter()
            .map(|(tok, &id)| (id, tok.as_str()))
            .collect();
        by_id.sort_by_key(|(id, _)| *id); // stable, ascending ID

        // Rebuild an ordered map purely for pretty, deterministic JSON
        // We’ll write as: [{"id":0,"token":"..."}, ...] to avoid key-order debates
        #[derive(serde::Serialize)]
        struct Row<'a> {
            id: u32,
            token: &'a str,
        }

        let rows: Vec<Row> = by_id
            .into_iter()
            .map(|(id, tok)| Row { id, token: tok })
            .collect();

        let mut file = File::create(path)?;
        let json = serde_json::to_string_pretty(&rows).expect("Serialization failed");
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load_vocab_id_list(path: &str) -> Result<HashMap<String, u32>> {
        use std::fs::File;
        use std::io::BufReader;

        #[derive(serde::Deserialize)]
        struct Row {
            id: u32,
            token: String,
        }

        let f = File::open(path)?;
        let r = BufReader::new(f);
        let rows: Vec<Row> = serde_json::from_reader(r)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut map = HashMap::with_capacity(rows.len());
        for Row { id, token } in rows {
            map.insert(token, id);
        }
        Ok(map)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn print_stats(&self) {
        println!("Total vocabulary size: {}", self.vocab.len());
        println!("Byte-level tokens (all 256 bytes, GPT-2 style): 256");
        println!("Tamil clustered characters: 247");
        println!("Tamil digits: 10");
        println!("Extra Tamil characters: 11");

        // Show mapping examples
        println!("\n=== Character Mapping Examples ===");
        println!(
            "Newline (\\n) → U+{:04X} ({})",
            self.byte_encoder[&b'\n'] as u32, self.byte_encoder[&b'\n']
        );
        println!(
            "Space ( ) → U+{:04X} ({})",
            self.byte_encoder[&b' '] as u32, self.byte_encoder[&b' ']
        );
        println!(
            "Tab (\\t) → U+{:04X} ({})",
            self.byte_encoder[&b'\t'] as u32, self.byte_encoder[&b'\t']
        );
    }
}
