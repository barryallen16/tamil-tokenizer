use std::sync::Arc;
use rustc_hash::FxHashMap;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, Anchored, Input, MatchKind, StartKind};
use crate::pretokenizer::TamilPretokenizer;
use crate::vocab::{TamilTokenizer, TokenizerError, Result};


#[derive(Debug, Clone)]
pub struct SpecialTokenConfig {
    pub core_tokens: Vec<String>,
    pub custom_tokens: Vec<String>,
}


impl Default for SpecialTokenConfig {
    fn default() -> Self {
        Self {
            core_tokens: vec![
                "<|startoftext|>".to_string(),
                "<|endoftext|>".to_string(),
                "<|pad|>".to_string(),
                "<|unk|>".to_string(),
                "<|mask|>".to_string(),
            ],
            custom_tokens: vec![
                "<|question|>".to_string(),
                "<|answer|>".to_string(),
                "<|code|>".to_string(),
            ],
        }
    }
}


#[allow(dead_code)]
pub struct FastBPEEncoder {
    token_to_id: Arc<FxHashMap<String, u32>>,
    token_to_id_bytes: Arc<FxHashMap<Vec<u8>, u32>>,
    id_to_token: Arc<Vec<String>>,
    ac_automaton: Arc<AhoCorasick>,
    pretokenizer: TamilPretokenizer,
    byte_encoder: FxHashMap<u8, char>,
    byte_decoder: FxHashMap<char, u8>,
    special_tokens: FxHashMap<String, u32>,
    special_token_ids: Vec<u32>,
    special_token_automaton: AhoCorasick,
    base_vocab_size: u32,
    named_tokens_start: u32,
    named_tokens_end: u32,
    reserved_start: u32,
    reserved_end: u32,
}


impl FastBPEEncoder {
    pub fn new(vocab: FxHashMap<String, u32>) -> Result<Self> {
        Self::new_with_config(vocab, SpecialTokenConfig::default())
    }


    pub fn new_with_config(mut vocab: FxHashMap<String, u32>, config: SpecialTokenConfig) -> Result<Self> {
        let base_vocab_size = vocab.values().max().copied().unwrap_or(0) + 1;
        
        let mut named_token_id = base_vocab_size;
        let named_tokens_start = named_token_id;
        
        let mut special_tokens = FxHashMap::default();
        let mut special_token_ids = Vec::new();
        let mut all_special_token_strings = Vec::new();
        
        for token in config.core_tokens {
            vocab.insert(token.clone(), named_token_id);
            special_tokens.insert(token.clone(), named_token_id);
            special_token_ids.push(named_token_id);
            all_special_token_strings.push(token.clone());
            named_token_id += 1;
        }
        
        for token in config.custom_tokens {
            vocab.insert(token.clone(), named_token_id);
            special_tokens.insert(token.clone(), named_token_id);
            special_token_ids.push(named_token_id);
            all_special_token_strings.push(token.clone());
            named_token_id += 1;
        }
        
        let named_tokens_end = named_token_id;
        
        let reserved_start = named_tokens_end;
        let reserved_count = 100;
        let reserved_end = reserved_start + reserved_count;
        
        for i in reserved_start..reserved_end {
            let reserved_token = format!("<|reserved_{}|>", i);
            vocab.insert(reserved_token.clone(), i as u32);
            all_special_token_strings.push(reserved_token);
        }

        let special_token_patterns: Vec<&str> = all_special_token_strings.iter()
            .map(|s| s.as_str())
            .collect();
        let special_token_automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&special_token_patterns)
            .map_err(|e| TokenizerError::InvalidData(format!("Failed to build special token automaton: {}", e)))?;

        let vocab_size = reserved_end as usize;
        let mut id_to_token = vec![String::new(); vocab_size];
        
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
        }

        // The vocab mixes raw-text tokens (Tamil clusters and merges, trained on
        // raw text) with 256 single-char byte tokens. Bytes outside printable
        // ASCII are stood in by chars U+0100.. which can also occur in real text,
        // so tokens containing a stand-in must never match raw input: they are
        // reachable only through the byte fallback in `encode_inner`.
        let (byte_encoder, byte_decoder) = TamilTokenizer::create_byte_mappings();
        let is_stand_in = |c: char| c as u32 >= 256 && byte_decoder.contains_key(&c);

        let patterns: Vec<&str> = vocab
            .keys()
            .filter(|k| !k.chars().any(is_stand_in))
            .map(|s| s.as_str())
            .collect();
        let ac_automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .start_kind(StartKind::Anchored)
            .build(&patterns)
            .map_err(|e| TokenizerError::InvalidData(format!("Failed to build Aho-Corasick automaton: {}", e)))?;

        let mut token_to_id_bytes: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for (token, &id) in &vocab {
            token_to_id_bytes.insert(token.as_bytes().to_vec(), id);
        }

        eprintln!("\n=== o200k-style Special Token Assignment ===");
        eprintln!("Base vocabulary size: {}", base_vocab_size);
        eprintln!("Named special tokens: {} to {} ({} tokens)",
                 named_tokens_start, named_tokens_end - 1,
                 named_tokens_end - named_tokens_start);
        eprintln!("Reserved tokens: {} to {} ({} tokens)",
                 reserved_start, reserved_end - 1, reserved_count);
        eprintln!("Total vocab size: {}\n", vocab_size);

        eprintln!("Named Special Tokens:");
        let mut special_sorted: Vec<_> = special_tokens.iter().collect();
        special_sorted.sort_by_key(|(_, id)| *id);
        for (token, id) in special_sorted {
            eprintln!("  {} = {}", id, token);
        }

        Ok(Self {
            token_to_id: Arc::new(vocab),
            token_to_id_bytes: Arc::new(token_to_id_bytes),
            id_to_token: Arc::new(id_to_token),
            ac_automaton: Arc::new(ac_automaton),
            pretokenizer: TamilPretokenizer::new(),
            byte_encoder: byte_encoder.into_iter().collect(),
            byte_decoder: byte_decoder.into_iter().collect(),
            special_tokens,
            special_token_ids,
            special_token_automaton,
            base_vocab_size,
            named_tokens_start,
            named_tokens_end,
            reserved_start,
            reserved_end: reserved_end as u32,
        })
    }

    fn encode_inner(&self, text: &str) -> Vec<u32> {
        let mut token_ids = Vec::with_capacity(text.len() / 3);
        let chunks = self.pretokenizer.pretokenize(text);

        for chunk in chunks {
            let mut pos = 0;

            while pos < chunk.len() {
                let input = Input::new(&chunk[pos..]).anchored(Anchored::Yes);
                if let Some(mat) = self.ac_automaton.find(input) {
                    // Patterns are valid UTF-8, so an anchored match ends on a
                    // char boundary.
                    let matched_str = &chunk[pos..pos + mat.end()];
                    if let Some(&id) = self.token_to_id.get(matched_str) {
                        token_ids.push(id);
                        pos += mat.end();
                        continue;
                    }
                }

                // No raw token covers this position: emit one byte token per
                // UTF-8 byte of the next char, so nothing is ever dropped.
                let ch = chunk[pos..].chars().next().unwrap();
                let mut buf = [0u8; 4];
                for &b in ch.encode_utf8(&mut buf).as_bytes() {
                    if let Some(&id) = self.token_to_id.get(&self.byte_encoder[&b].to_string()) {
                        token_ids.push(id);
                    }
                }
                pos += ch.len_utf8();
            }
        }

        token_ids
    }

    /// Encode with BOS/EOS, but DO NOT auto-detect special tokens
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        result.push(self.special_tokens["<|startoftext|>"]);
        result.extend(self.encode_inner(text));
        result.push(self.special_tokens["<|endoftext|>"]);
        result
    }

    /// Encode ordinary (like tiktoken playground)
    /// Returns ONLY text tokens - recognizes special tokens if in vocab
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.encode_inner(text)
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.decode_inner(token_ids)
    }

    pub fn decode_skip_special(&self, token_ids: &[u32]) -> String {
        let filtered: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&id| !self.is_special_token(id))
            .collect();
        self.decode_inner(&filtered)
    }

    /// Reverse of `encode_inner`: join token strings, map every byte-mapped char
    /// back to its raw byte, then reinterpret those bytes as UTF-8.
    fn decode_inner(&self, token_ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(token_ids.len() * 4);
        for &id in token_ids {
            let Some(token) = self.id_to_token.get(id as usize) else { continue };
            for ch in token.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn is_special_token(&self, token_id: u32) -> bool {
        token_id >= self.named_tokens_start && token_id < self.reserved_end
    }

    pub fn get_special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn special_token_count(&self) -> usize {
        self.special_tokens.len()
    }

    pub fn reserved_token_count(&self) -> u32 {
        self.reserved_end - self.named_tokens_end
    }
}
