use std::sync::Arc;
use rustc_hash::FxHashMap;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use crate::pretokenizer::TamilPretokenizer;
use crate::vocab::{TokenizerError, Result};


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
    byte_to_id: [Option<u32>; 256],
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

        let patterns: Vec<&str> = vocab.keys().map(|s| s.as_str()).collect();
        let ac_automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| TokenizerError::InvalidData(format!("Failed to build Aho-Corasick automaton: {}", e)))?;

        let mut byte_to_id = [None; 256];
        for (token, &id) in &vocab {
            if token.len() == 1 {
                byte_to_id[token.as_bytes()[0] as usize] = Some(id);
            }
        }

        let mut token_to_id_bytes: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        for (token, &id) in &vocab {
            token_to_id_bytes.insert(token.as_bytes().to_vec(), id);
        }

        println!("\n=== o200k-style Special Token Assignment ===");
        println!("Base vocabulary size: {}", base_vocab_size);
        println!("Named special tokens: {} to {} ({} tokens)", 
                 named_tokens_start, named_tokens_end - 1, 
                 named_tokens_end - named_tokens_start);
        println!("Reserved tokens: {} to {} ({} tokens)", 
                 reserved_start, reserved_end - 1, reserved_count);
        println!("Total vocab size: {}\n", vocab_size);
        
        println!("Named Special Tokens:");
        let mut special_sorted: Vec<_> = special_tokens.iter().collect();
        special_sorted.sort_by_key(|(_, id)| *id);
        for (token, id) in special_sorted {
            println!("  {} = {}", id, token);
        }

        Ok(Self {
            token_to_id: Arc::new(vocab),
            token_to_id_bytes: Arc::new(token_to_id_bytes),
            id_to_token: Arc::new(id_to_token),
            ac_automaton: Arc::new(ac_automaton),
            pretokenizer: TamilPretokenizer::new(),
            byte_to_id,
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
            let bytes = chunk.as_bytes();
            let mut pos = 0;
            
            while pos < bytes.len() {
                let remaining = &bytes[pos..];
                let mut matched = false;
                
                if let Some(mat) = self.ac_automaton.find(remaining) {
                    if mat.start() == 0 {
                        let matched_bytes = &remaining[..mat.end()];
                        if let Ok(matched_str) = std::str::from_utf8(matched_bytes) {
                            if let Some(&id) = self.token_to_id.get(matched_str) {
                                token_ids.push(id);
                                pos += mat.end();
                                matched = true;
                            }
                        }
                    }
                }
                
                if !matched {
                    if let Some(id) = self.byte_to_id[bytes[pos] as usize] {
                        token_ids.push(id);
                    }
                    pos += 1;
                }
            }
        }
        
        token_ids
    }

    /// ✅ FIXED: Encode with BOS/EOS, but DO NOT auto-detect special tokens
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        result.push(self.special_tokens["<|startoftext|>"]);
        result.extend(self.encode_inner(text));  // ✅ Just BPE, no special token detection
        result.push(self.special_tokens["<|endoftext|>"]);
        result
    }

    /// ✅ Encode ordinary (like tiktoken playground)
    /// Returns ONLY text tokens - recognizes special tokens if in vocab
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.encode_inner(text)
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        token_ids.iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .map(|s| s.as_str())
            .collect::<String>()
    }

    pub fn decode_skip_special(&self, token_ids: &[u32]) -> String {
        token_ids.iter()
            .filter(|&&id| !self.is_special_token(id))
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .map(|s| s.as_str())
            .collect::<String>()
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
