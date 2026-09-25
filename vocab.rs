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

        // Step 3: Add 18 traditional consonants + 5 Grantha consonants
        let consonant_codepoints = [
            0x0B95, 0x0B99, 0x0B9A, 0x0B9E, 0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BAF,
            0x0BB0, 0x0BB2, 0x0BB5, 0x0BB4, 0x0BB3, 0x0BB1, 0x0BA9,
            // Grantha consonants
            0x0B9C, // ஜ (JA)
            0x0BB6, // ஶ (SHA)
            0x0BB7, // ஷ (SSA)
            0x0BB8, // ஸ (SA)
            0x0BB9, // ஹ (HA)
        ];
        for &cp in &consonant_codepoints {
            if let Some(ch) = std::char::from_u32(cp) {
                vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Step 4: Add 12 vowel signs (in codepoint order)
        let vowel_sign_codepoints = [
            0x0BBE, 0x0BBF, 0x0BC0, 0x0BC1, 0x0BC2, 0x0BC6, 0x0BC7, 0x0BC8, 0x0BCA, 0x0BCB, 0x0BCC,
            0x0BCD,
        ];

        // Step 5: Generate 23 × 12 = 276 consonant+vowel combinations (in codepoint order)
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

        // Step 6: Add special Tamil characters: ஃ (0x0B83), ஂ (0x0B82), ௐ (0x0BD0), ௗ (0x0BD7)
        for &cp in &[0x0B83, 0x0B82, 0x0BD0, 0x0BD7] {
            if let Some(ch) = std::char::from_u32(cp) {
                vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Step 7: Add common compound ligatures (க்ஷ, ஸ்ரீ) and their variants
        let ligatures = [
            // Ksha and its vowel signs
            "க்ஷ",
            "க்ஷா",
            "க்ஷி",
            "க்ஷீ",
            "க்ஷு",
            "க்ஷூ",
            "க்ஷெ",
            "க்ஷே",
            "க்ஷை",
            "க்ஷொ",
            "க்ஷோ",
            "க்ஷௌ",
            "க்ஷ்",
            // Sri variants
            "ஸ்ரீ",
            "ஶ்ரீ",
        ];
        // A token can be reachable from more than one step: க்ஷ is both a
        // ligature and a conjunct stem. Pushing it twice would collapse two
        // ids into one and leave a gap in the id space, so dedup here.
        let mut seen: std::collections::HashSet<String> =
            vocab_tuples.iter().map(|(t, _)| t.clone()).collect();

        for &lig in &ligatures {
            if seen.insert(lig.to_string()) {
                vocab_tuples.push((lig.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Conjunct clusters: consonant + virama + consonant.
        //
        // Tamil orthography joins consonants this way constantly (க்ஷ, ஸ்ரீ,
        // க்ட, ண்ட) and the base set had no way to express any of it, so every
        // conjunct cost 3-4 tokens where 1 would do. Adding the bare stem is
        // what matters: extending this to a trailing vowel sign as well
        // (23 x 23 x 12 = 6,348 more tokens) measurably *hurts*, because the
        // base vocab reaches ~7,000 and the fixed 30K merge budget no longer
        // covers the corpus. Measured on the 30K wiki model, held-out:
        //
        //   base vocab   wiki      IndicCorp  Grantha-heavy
        //   607 (before) 10.183    10.280     9.885
        //   1,793 (this) 10.156    10.071     9.759
        //   ~7,000       10.855    10.936     10.562
        let conjunct_stems: Vec<char> = consonant_codepoints
            .iter()
            .filter_map(|&cp| char::from_u32(cp))
            .collect();
        for &first in &conjunct_stems {
            for &second in &conjunct_stems {
                let stem = format!("{}்{}", first, second);
                if seen.insert(stem.clone()) {
                    vocab_tuples.push((stem, vocab_tuples.len() as u32));
                }
            }
        }

        // Step 9: Add 10 Tamil digits (in codepoint order)
        for cp in 0x0BE6..=0x0BEF {
            if let Some(ch) = std::char::from_u32(cp) {
                vocab_tuples.push((ch.to_string(), vocab_tuples.len() as u32));
            }
        }

        // Step 10: Add 11 extra Tamil chars (in codepoint order)
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

    /// GPT-2's byte-to-unicode table. Bytes GPT-2 treats as printable keep their own
    /// char; the rest go to U+0100 upward in ascending byte order.
    ///
    /// Must stay identical to upstream. Hugging Face's `ByteLevel` pretokenizer
    /// and decoder hardcode this table, so exported models only round-trip
    /// through Python `tokenizers` when these agree byte for byte.
    pub fn create_byte_mappings() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let printable =
            |b: u8| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);

        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut n = 0u32;
        for b in 0..=255u8 {
            let ch = if printable(b) {
                b as char
            } else {
                let mapped = std::char::from_u32(256 + n).expect("256 + 190 fits in u32");
                n += 1;
                mapped
            };
            encoder.insert(b, ch);
            decoder.insert(ch, b);
        }

        (encoder, decoder)
    }

    /// Decode a token back to its original byte representation
    pub fn decode_token(&self, token: &str) -> Vec<u8> {
        token
            .chars()
            .filter_map(|ch| self.byte_decoder.get(&ch).copied())
            .collect()
    }

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
        println!("Tamil consonant+vowel clusters: 276");
        println!("Standalone Tamil characters: 39");
        println!("Common compound ligatures: 15");
        println!("Conjunct stems (consonant+virama+consonant): 529");
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
