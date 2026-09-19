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

#[derive(Clone, Copy, Debug)]
struct Symbol {
    start: usize,
    end: usize,
    id: Option<u32>,
}

#[inline]
fn pack_pair(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

pub struct FastBPEEncoder {
    token_to_id: Arc<FxHashMap<String, u32>>,
    id_to_token: Arc<Vec<String>>,
    base_automaton: Arc<AhoCorasick>,
    pattern_to_id: Arc<Vec<u32>>,
    pair_to_merge: FxHashMap<u64, (u32, u32)>,
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
        let max_id = vocab.values().max().copied();
        let max_possible = vocab
            .len()
            .saturating_add(config.core_tokens.len())
            .saturating_add(config.custom_tokens.len())
            .saturating_add(100);

        if let Some(max) = max_id {
            if max as usize >= max_possible {
                return Err(TokenizerError::InvalidData(format!(
                    "token id {} is out of range for vocab of size {}: ids must be contiguous from 0",
                    max,
                    vocab.len()
                )));
            }
        }

        let base_vocab_size = match max_id {
            Some(max) => max.checked_add(1).ok_or_else(|| {
                TokenizerError::InvalidData("token ID causes overflow".to_string())
            })?,
            None => 0,
        };

        // Reuse IDs already in the file so a trained tokenizer keeps its IDs.
        // Only genuinely missing tokens are appended after the current max.
        let mut next_id = base_vocab_size;
        let mut claim = |vocab: &mut FxHashMap<String, u32>, token: String| -> Result<u32> {
            if let Some(&id) = vocab.get(&token) {
                return Ok(id);
            }
            let id = next_id;
            next_id = next_id.checked_add(1).ok_or_else(|| {
                TokenizerError::InvalidData("vocab size exceeds u32 limit".to_string())
            })?;
            vocab.insert(token, id);
            Ok(id)
        };

        let mut special_tokens = FxHashMap::default();
        let mut special_token_ids = Vec::new();

        for token in config.core_tokens.into_iter().chain(config.custom_tokens) {
            let id = claim(&mut vocab, token.clone())?;
            special_tokens.insert(token, id);
            special_token_ids.push(id);
        }

        let named_tokens_start = special_token_ids.iter().copied().min().unwrap_or(next_id);
        let named_tokens_end = special_token_ids
            .iter()
            .copied()
            .max()
            .and_then(|m| m.checked_add(1))
            .unwrap_or(next_id);

        let reserved_count = 100usize;
        let existing_reserved = vocab
            .keys()
            .filter(|k| k.starts_with("<|reserved_"))
            .count();
        for _ in 0..reserved_count.saturating_sub(existing_reserved) {
            let id = next_id;
            next_id = next_id.checked_add(1).ok_or_else(|| {
                TokenizerError::InvalidData("vocab size exceeds u32 limit".to_string())
            })?;
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
            .and_then(|m| m.checked_add(1))
            .unwrap_or(next_id)
            .max(named_tokens_end);

        let vocab_size = next_id as usize;
        let mut id_to_token = vec![String::new(); vocab_size];

        // A vocab whose IDs are not dense 0..vocab_size would either panic here
        // or silently decode to the wrong token, so reject it up front. This also
        // stops a huge ID from sizing the vector into an allocation bomb.
        for (token, &id) in &vocab {
            if id as usize >= vocab_size {
                return Err(TokenizerError::InvalidData(format!(
                    "token {:?} has id {} but vocab_size is {}: ids must be \
                     contiguous from 0",
                    token, id, vocab_size
                )));
            }
            id_to_token[id as usize] = token.clone();
        }

        if id_to_token.iter().any(String::is_empty) {
            return Err(TokenizerError::InvalidData(
                "vocab has a gap in its ids; ids must be contiguous from 0".to_string(),
            ));
        }

        // The vocab mixes raw-text tokens (Tamil clusters and merges, trained on
        // raw text) with 256 single-char byte tokens. Bytes outside printable
        // ASCII are stood in by chars U+0100.. which can also occur in real text,
        // so tokens containing a stand-in must never match raw input: they are
        // reachable only through the byte fallback in `encode_inner`.
        let (byte_encoder, byte_decoder) = TamilTokenizer::create_byte_mappings();
        // A stand-in char is any byte token that raw text must never match
        // directly. Under GPT-2's table bytes 161..=255 map to ordinary Latin-1
        // chars, so `Ã` is both a byte token and real text; only printable ASCII
        // is safe to match in the automaton, because for those the byte token
        // and the raw char really are the same token.
        let is_stand_in = |c: char| {
            byte_decoder.contains_key(&c) && !(33..=126).contains(&(c as u32))
        };

        // Merge results are excluded from base matching so rank order decides.
        // Rank = position in the trainer's merge list (0 = learned first).
        let mut pair_to_merge: FxHashMap<u64, (u32, u32)> = FxHashMap::default();
        let mut merged_strings = std::collections::HashSet::new();
        for (rank, (a, b)) in merges.iter().enumerate() {
            let merged = format!("{}{}", a, b);
            if let (Some(&id_a), Some(&id_b), Some(&new_id)) =
                (vocab.get(a), vocab.get(b), vocab.get(&merged))
            {
                let key = pack_pair(id_a, id_b);
                pair_to_merge.insert(key, (rank as u32, new_id));
            }
            merged_strings.insert(merged);
        }

        let base_patterns: Vec<&str> = vocab
            .keys()
            .filter(|k| !k.chars().any(is_stand_in) && !merged_strings.contains(k.as_str()))
            .map(|s| s.as_str())
            .collect();
        let pattern_to_id: Vec<u32> = base_patterns.iter().map(|&p| vocab[p]).collect();
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
            pattern_to_id: Arc::new(pattern_to_id),
            pair_to_merge,
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

        for chunk in chunks {
            let mut symbols = self.segment(chunk);
            self.apply_merges(&mut symbols);
            for sym in &symbols {
                if let Some(id) = sym.id {
                    token_ids.push(id);
                } else {
                    self.push_text(&mut token_ids, &chunk[sym.start..sym.end]);
                }
            }
        }

        token_ids
    }

    /// Longest-match over base tokens only (merge results excluded so rank
    /// order decides). Unmatched positions become single-char spans.
    /// Spans are byte ranges, so segmentation allocates nothing.
    fn segment(&self, chunk: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let mut pos = 0;

        while pos < chunk.len() {
            let input = Input::new(&chunk[pos..]).anchored(Anchored::Yes);
            if let Some(mat) = self.base_automaton.find(input) {
                let id = self.pattern_to_id[mat.pattern().as_usize()];
                symbols.push(Symbol {
                    start: pos,
                    end: pos + mat.end(),
                    id: Some(id),
                });
                pos += mat.end();
            } else {
                let ch = chunk[pos..].chars().next().unwrap();
                let len = ch.len_utf8();
                let mut buf = [0u8; 4];
                let single = ch.encode_utf8(&mut buf);
                let id = if !self.is_byte_stand_in(ch) {
                    self.token_to_id.get(single).copied()
                } else {
                    None
                };
                symbols.push(Symbol {
                    start: pos,
                    end: pos + len,
                    id,
                });
                pos += len;
            }
        }

        symbols
    }

    /// Standard BPE using fast packed integer lookups in pair_to_merge: repeatedly
    /// merge the adjacent pair with the lowest rank (earliest learned); ties go
    /// to the leftmost pair.
    fn apply_merges(&self, symbols: &mut Vec<Symbol>) {
        if self.pair_to_merge.is_empty() || symbols.len() < 2 {
            return;
        }
        while symbols.len() >= 2 {
            let mut best: Option<(usize, u32, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if let (Some(id_a), Some(id_b)) = (symbols[i].id, symbols[i + 1].id) {
                    let key = pack_pair(id_a, id_b);
                    if let Some(&(rank, new_id)) = self.pair_to_merge.get(&key) {
                        if best.is_none_or(|(_, r, _)| rank < r) {
                            best = Some((i, rank, new_id));
                        }
                    }
                }
            }
            match best {
                Some((i, _, new_id)) => {
                    symbols[i].end = symbols[i + 1].end;
                    symbols[i].id = Some(new_id);
                    symbols.remove(i + 1);
                }
                None => break,
            }
        }
    }

    fn is_byte_stand_in(&self, c: char) -> bool {
        self.byte_decoder.contains_key(&c) && !(33..=126).contains(&(c as u32))
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

    /// Parallel batch encoding with BOS/EOS using rayon
    pub fn encode_batch<T: AsRef<str> + Sync>(&self, texts: &[T]) -> Vec<Vec<u32>> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.encode(t.as_ref())).collect()
    }

    /// Parallel batch encoding without special tokens using rayon
    pub fn encode_ordinary_batch<T: AsRef<str> + Sync>(&self, texts: &[T]) -> Vec<Vec<u32>> {
        use rayon::prelude::*;
        texts
            .par_iter()
            .map(|t| self.encode_ordinary(t.as_ref()))
            .collect()
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
