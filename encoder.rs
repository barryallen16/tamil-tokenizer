use crate::pretokenizer::TamilPretokenizer;
use crate::vocab::{Result, TamilTokenizer, TokenizerError};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, Anchored, Input, MatchKind, StartKind};
use rustc_hash::FxHashMap;
use std::sync::Arc;

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

pub struct FastBPEEncoder {
    token_to_id: Arc<FxHashMap<String, u32>>,
    id_to_token: Arc<Vec<String>>,
    base_automaton: Arc<AhoCorasick>,
    merge_ranks: FxHashMap<String, FxHashMap<String, u32>>,
    pretokenizer: TamilPretokenizer,
    byte_encoder: FxHashMap<u8, char>,
    byte_decoder: FxHashMap<char, u8>,
    special_tokens: FxHashMap<String, u32>,
    named_tokens_start: u32,
    named_tokens_end: u32,
    reserved_end: u32,
}

impl FastBPEEncoder {
    pub fn new(vocab: FxHashMap<String, u32>) -> Result<Self> {
        Self::new_with_config(vocab, SpecialTokenConfig::default())
    }

    pub fn new_with_config(
        vocab: FxHashMap<String, u32>,
        config: SpecialTokenConfig,
    ) -> Result<Self> {
        Self::build(vocab, config, &[])
    }

    /// Build from a trained tokenizer file: vocab plus ordered merge pairs.
    /// Merge order is applied at encode time (true BPE); without merges the
    /// encoder falls back to longest-match over the vocab.
    pub fn new_with_merges(
        vocab: FxHashMap<String, u32>,
        merges: &[(String, String)],
    ) -> Result<Self> {
        Self::build(vocab, SpecialTokenConfig::default(), merges)
    }

    fn build(
        mut vocab: FxHashMap<String, u32>,
        config: SpecialTokenConfig,
        merges: &[(String, String)],
    ) -> Result<Self> {
        let base_vocab_size = vocab.values().max().copied().unwrap_or(0) + 1;

        // Reuse IDs already in the file so a trained tokenizer keeps its IDs.
        // Only genuinely missing tokens are appended after the current max.
        let mut next_id = base_vocab_size;
        let mut claim = |vocab: &mut FxHashMap<String, u32>, token: String| -> u32 {
            if let Some(&id) = vocab.get(&token) {
                return id;
            }
            let id = next_id;
            next_id += 1;
            vocab.insert(token, id);
            id
        };

        let mut special_tokens = FxHashMap::default();
        let mut special_token_ids = Vec::new();

        for token in config.core_tokens.into_iter().chain(config.custom_tokens) {
            let id = claim(&mut vocab, token.clone());
            special_tokens.insert(token, id);
            special_token_ids.push(id);
        }

        let named_tokens_start = special_token_ids.iter().copied().min().unwrap_or(next_id);
        let named_tokens_end = special_token_ids
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(next_id);

        let reserved_count = 100usize;
        let existing_reserved = vocab
            .keys()
            .filter(|k| k.starts_with("<|reserved_"))
            .count();
        for _ in 0..reserved_count.saturating_sub(existing_reserved) {
            let id = next_id;
            next_id += 1;
            let reserved_token = format!("<|reserved_{}|>", id);
            vocab.insert(reserved_token, id);
        }

        let reserved_ids: Vec<u32> = vocab
            .iter()
            .filter(|(k, _)| k.starts_with("<|reserved_"))
            .map(|(_, &id)| id)
            .collect();
        let reserved_end = reserved_ids
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(next_id)
            .max(named_tokens_end);

        let vocab_size = next_id as usize;
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

        // Merge results are excluded from base matching so rank order decides.
        // Rank = position in the trainer's merge list (0 = learned first).
        let mut merge_ranks: FxHashMap<String, FxHashMap<String, u32>> = FxHashMap::default();
        let mut merged_strings = std::collections::HashSet::new();
        for (rank, (a, b)) in merges.iter().enumerate() {
            merge_ranks
                .entry(a.clone())
                .or_default()
                .insert(b.clone(), rank as u32);
            merged_strings.insert(format!("{}{}", a, b));
        }

        let base_patterns: Vec<&str> = vocab
            .keys()
            .filter(|k| !k.chars().any(is_stand_in) && !merged_strings.contains(k.as_str()))
            .map(|s| s.as_str())
            .collect();
        let base_automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .start_kind(StartKind::Anchored)
            .build(&base_patterns)
            .map_err(|e| {
                TokenizerError::InvalidData(format!(
                    "Failed to build Aho-Corasick automaton: {}",
                    e
                ))
            })?;

        Ok(Self {
            token_to_id: Arc::new(vocab),
            id_to_token: Arc::new(id_to_token),
            base_automaton: Arc::new(base_automaton),
            merge_ranks,
            pretokenizer: TamilPretokenizer::new(),
            byte_encoder: byte_encoder.into_iter().collect(),
            byte_decoder: byte_decoder.into_iter().collect(),
            special_tokens,
            named_tokens_start,
            named_tokens_end,
            reserved_end,
        })
    }

    fn encode_inner(&self, text: &str) -> Vec<u32> {
        let mut token_ids = Vec::with_capacity(text.len() / 3);
        let chunks = self.pretokenizer.pretokenize(text);

        for chunk in &chunks {
            let mut symbols = self.segment(chunk);
            self.apply_merges(chunk, &mut symbols);
            for (a, b) in &symbols {
                self.push_text(&mut token_ids, &chunk[*a..*b]);
            }
        }

        token_ids
    }

    /// Longest-match over base tokens only (merge results excluded so rank
    /// order decides). Unmatched positions become single-char spans.
    /// Spans are byte ranges, so segmentation allocates nothing.
    fn segment(&self, chunk: &str) -> Vec<(usize, usize)> {
        let mut symbols = Vec::new();
        let mut pos = 0;

        while pos < chunk.len() {
            let input = Input::new(&chunk[pos..]).anchored(Anchored::Yes);
            if let Some(mat) = self.base_automaton.find(input) {
                symbols.push((pos, pos + mat.end()));
                pos += mat.end();
            } else {
                let len = chunk[pos..].chars().next().unwrap().len_utf8();
                symbols.push((pos, pos + len));
                pos += len;
            }
        }

        symbols
    }

    /// Standard BPE: repeatedly merge the adjacent pair with the lowest rank
    /// (earliest learned). Each merge shortens the sequence, so this ends.
    fn apply_merges(&self, chunk: &str, symbols: &mut Vec<(usize, usize)>) {
        if self.merge_ranks.is_empty() {
            return;
        }
        while symbols.len() >= 2 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                let a = &chunk[symbols[i].0..symbols[i].1];
                let b = &chunk[symbols[i + 1].0..symbols[i + 1].1];
                if let Some(&rank) = self.merge_ranks.get(a).and_then(|m| m.get(b)) {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            match best {
                Some((i, _)) => {
                    let (_, end) = symbols.remove(i + 1);
                    symbols[i].1 = end;
                }
                None => break,
            }
        }
    }

    fn is_byte_stand_in(&self, c: char) -> bool {
        c as u32 >= 256 && self.byte_decoder.contains_key(&c)
    }

    /// Resolve a span to token IDs. Anything without an ID falls back to one
    /// token per UTF-8 byte, so nothing is ever dropped.
    fn push_text(&self, token_ids: &mut Vec<u32>, s: &str) {
        if !s.chars().any(|c| self.is_byte_stand_in(c)) {
            if let Some(&id) = self.token_to_id.get(s) {
                token_ids.push(id);
                return;
            }
        }
        for ch in s.chars() {
            if !self.is_byte_stand_in(ch) {
                let mut buf = [0u8; 4];
                let single = ch.encode_utf8(&mut buf);
                if let Some(&id) = self.token_to_id.get(single) {
                    token_ids.push(id);
                    continue;
                }
            }
            let mut buf = [0u8; 4];
            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                if let Some(&id) = self.token_to_id.get(&self.byte_encoder[&b].to_string()) {
                    token_ids.push(id);
                }
            }
        }
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
            let Some(token) = self.id_to_token.get(id as usize) else {
                continue;
            };
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
