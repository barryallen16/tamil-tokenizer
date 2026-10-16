use crate::model::{canonical_model, parse_byte_token};
use crate::pretokenizer::TamilPretokenizer;
use crate::vocab::{Result, TokenizerError};
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

/// Merge rules in CSR layout, keyed by the first token of the pair.
///
/// A hash map was the obvious choice, and it is what this used to be. At a
/// 120K-merge vocab it cost 2.4x throughput (17.4 -> 7.3 MB/s) while the
/// automaton over the same 122K patterns cost only 17%: the table stops
/// fitting in cache and every probe misses.
///
/// A BPE merge table is almost all empty though. Each token is the *result* of
/// at most one merge, so the pairs starting with a given id are few and
/// contiguous when sorted. Indexing by `a` and binary-searching a short slice
/// replaces the random probe with one sequential read plus a couple of
/// comparisons.
/// Pack the second token, the rank and the merged token ID into one word.
#[inline]
fn pack(b: u32, rank: u32, new_id: u32) -> u64 {
    (b as u64) | ((rank as u64) << 21) | ((new_id as u64) << 42)
}

#[inline]
fn unpack_b(entry: u64) -> u32 {
    (entry & 0x1F_FFFF) as u32
}

#[inline]
fn unpack_rank(entry: u64) -> u32 {
    ((entry >> 21) & 0x1F_FFFF) as u32
}

#[inline]
fn unpack_new_id(entry: u64) -> u32 {
    ((entry >> 42) & 0x1F_FFFF) as u32
}

struct MergeTable {
    /// `offsets[a]..offsets[a + 1]` is the slice of `data` for first token `a`.
    offsets: Vec<u32>,
    /// Sorted by `b` within each slice. Each entry packs the second token,
    /// the rank and the merged token ID into one word: 17 bits each, which
    /// fits in 64. The table is read once per adjacent pair per merge, so
    /// halving its size halves the cache misses.
    data: Vec<u64>,
    /// One bit per token ID, set when that ID starts at least one merge.
    /// Most adjacent pairs in a chunk have no rule at all, and this answers
    /// that without touching the two large arrays.
    starts: Vec<u64>,
}

impl MergeTable {
    fn build(merges: &[(String, String)], ids: &FxHashMap<String, u32>, vocab_size: usize) -> Self {
        let mut pairs: Vec<(u32, u32, u32, u32)> = merges
            .iter()
            .enumerate()
            .filter_map(|(rank, (a, b))| {
                let merged = format!("{a}{b}");
                match (ids.get(a), ids.get(b), ids.get(&merged)) {
                    (Some(&ia), Some(&ib), Some(&new_id)) => Some((ia, ib, rank as u32, new_id)),
                    _ => None,
                }
            })
            .collect();
        pairs.sort_unstable_by_key(|&(a, b, ..)| (a, b));

        let mut offsets = vec![0u32; vocab_size + 1];
        for &(a, ..) in &pairs {
            offsets[(a + 1) as usize] += 1;
        }
        for i in 0..vocab_size {
            offsets[i + 1] += offsets[i];
        }

        let mut starts = vec![0u64; (vocab_size + 63) / 64];
        for &(a, ..) in &pairs {
            starts[(a / 64) as usize] |= 1u64 << (a % 64);
        }

        let mut data = vec![0u64; pairs.len()];
        let mut cursor = offsets.clone();
        for (a, b, rank, new_id) in pairs {
            let slot = &mut cursor[a as usize];
            data[*slot as usize] = pack(b, rank, new_id);
            *slot += 1;
        }

        Self {
            offsets,
            data,
            starts,
        }
    }

    /// Whether any merge starts with this token.
    #[inline]
    fn starts_with(&self, a: u32) -> bool {
        self.starts[(a / 64) as usize] & (1u64 << (a % 64)) != 0
    }

    #[inline]
    fn get(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        if !self.starts_with(a) {
            return None;
        }
        let (start, end) = (
            self.offsets[a as usize] as usize,
            self.offsets[(a + 1) as usize] as usize,
        );
        let slice = &self.data[start..end];
        // Slices are short; a plain scan beats binary search on branch
        // prediction for the 0-3 element case that dominates.
        for &entry in slice {
            let b2 = unpack_b(entry);
            if b2 == b {
                return Some((unpack_rank(entry), unpack_new_id(entry)));
            }
            if b2 > b {
                return None;
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct FastBPEEncoder {
    token_to_id: Arc<FxHashMap<String, u32>>,
    id_to_token: Arc<Vec<String>>,
    merges: MergeTable,
    /// The promotion merges, which rank ahead of the learned ones.
    promotions: MergeTable,
    pretokenizer: TamilPretokenizer,
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

        // The canonical model: the trained vocab plus the characters it uses
        // but does not hold, the byte tokens, and the promotion merges. The
        // Hugging Face export builds the same thing, so both engines read one
        // vocabulary and one merge list and emit the same IDs.
        let canonical = canonical_model(&vocab, merges);
        let vocab = canonical.vocab;
        let vocab_size = vocab.len();
        let mut id_to_token = vec![String::new(); vocab_size];
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
        }

        // Rank is position in the merge list. The promotion merges come first,
        // so a base cluster exists before any learned merge that consumes it.
        //
        // They are kept in a table of their own. There are a few hundred of
        // them and they run against single characters, so the whole table fits
        // in cache, while the learned merges do not.
        let (promotion_merges, learned_merges) = canonical.merges.split_at(canonical.promotions);
        let promotion_table = MergeTable::build(promotion_merges, &vocab, vocab_size);
        let merge_table = MergeTable::build(learned_merges, &vocab, vocab_size);


        Ok(Self {
            token_to_id: Arc::new(vocab),
            id_to_token: Arc::new(id_to_token),
            merges: merge_table,
            promotions: promotion_table,
            pretokenizer: TamilPretokenizer::new(),
            special_tokens,
            named_tokens_start,
            named_tokens_end,
            reserved_end,
        })
    }

    fn encode_inner(&self, text: &str) -> Vec<u32> {
        let mut token_ids = Vec::with_capacity(text.len() / 3);
        let chunks = self.pretokenizer.pretokenize(text);

        // The pretokenizer averages ~19 bytes per chunk, so a fresh Vec per
        // chunk is one allocation per handful of bytes. Reuse one scratch
        // buffer across the whole text instead.
        let mut symbols = Vec::with_capacity(64);
        let mut cache: Vec<Option<(u32, u32)>> = Vec::with_capacity(64);
        for chunk in chunks {
            symbols.clear();
            self.seed_chars(chunk, &mut symbols);
            self.apply_merges(&mut symbols, &mut cache);
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

    /// Seed a chunk with one symbol per character.
    ///
    /// Hugging Face's `BPE` model has no seed symbols, so this is the only
    /// starting point it can express. The base clusters come back through the
    /// promotion merges, which rank ahead of the learned ones.
    ///
    /// A special token is a single symbol, because Hugging Face extracts added
    /// tokens before the BPE model runs.
    fn seed_chars(&self, chunk: &str, symbols: &mut Vec<Symbol>) {
        symbols.clear();
        if let Some(&id) = self.special_tokens.get(chunk) {
            symbols.push(Symbol {
                start: 0,
                end: chunk.len(),
                id: Some(id),
            });
            return;
        }
        for (offset, ch) in chunk.char_indices() {
            let mut buf = [0u8; 4];
            let single = ch.encode_utf8(&mut buf);
            let id = self.token_to_id.get(single).copied();
            symbols.push(Symbol {
                start: offset,
                end: offset + ch.len_utf8(),
                id,
            });
        }
    }

    /// Standard BPE using fast packed integer lookups in pair_to_merge: repeatedly
    /// merge the adjacent pair with the lowest rank (earliest learned); ties go
    /// to the leftmost pair.
    ///
    /// Every occurrence of that pair merges in one pass, which is what
    /// Hugging Face does. Merging one occurrence at a time segments the same
    /// text differently once a merge creates a pair that outranks the next
    /// candidate.
    fn apply_merges(&self, symbols: &mut Vec<Symbol>, cache: &mut Vec<Option<(u32, u32)>>) {
        // The promotion merges rank ahead of the learned ones, so running them
        // first is the same as running one list in rank order.
        self.merge_pass(symbols, cache, &self.promotions);
        self.merge_pass(symbols, cache, &self.merges);
    }

    /// Merge until no pair has a rule, taking the lowest rank each time.
    ///
    /// `cache` holds the rule for each adjacent pair. A pair keeps its rule
    /// until one of its symbols changes, so a pass over a chunk looks up each
    /// pair once rather than once per pass. Without it, a chunk that merges
    /// into a single token costs one table lookup per pair per pass, and the
    /// table is too large to stay in cache.
    fn merge_pass(
        &self,
        symbols: &mut Vec<Symbol>,
        cache: &mut Vec<Option<(u32, u32)>>,
        table: &MergeTable,
    ) {
        if table.len() == 0 || symbols.len() < 2 {
            return;
        }
        cache.clear();
        cache.resize(symbols.len() - 1, None);
        while symbols.len() >= 2 {
            let mut best: Option<(usize, u32, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if cache[i].is_none() {
                    cache[i] = match (symbols[i].id, symbols[i + 1].id) {
                        (Some(id_a), Some(id_b)) => table.get(id_a, id_b),
                        _ => None,
                    };
                }
                if let Some((rank, new_id)) = cache[i] {
                    if best.is_none_or(|(_, r, _)| rank < r) {
                        best = Some((i, rank, new_id));
                    }
                }
            }
            let Some((pos, _, new_id)) = best else {
                break;
            };
            let (Some(id_a), Some(id_b)) = (symbols[pos].id, symbols[pos + 1].id) else {
                break;
            };
            let mut i = 0;
            while i + 1 < symbols.len() {
                if symbols[i].id == Some(id_a) && symbols[i + 1].id == Some(id_b) {
                    symbols[i].end = symbols[i + 1].end;
                    symbols[i].id = Some(new_id);
                    symbols.remove(i + 1);
                    cache.remove(i);
                    // The pair that follows the merge is new, and so is the one
                    // before it.
                    if i < cache.len() {
                        cache[i] = None;
                    }
                    if i > 0 {
                        cache[i - 1] = None;
                    }
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Resolve a span to token IDs. A character the vocab does not hold falls
    /// back to one token per UTF-8 byte, so nothing is ever dropped.
    fn push_text(&self, token_ids: &mut Vec<u32>, s: &str) {
        for &b in s.as_bytes() {
            let name = format!("<0x{b:02X}>");
            if let Some(&id) = self.token_to_id.get(&name) {
                token_ids.push(id);
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

    /// Reverse of `encode_inner`: a `<0xNN>` token contributes its byte, and
    /// every other token contributes its text. Then the bytes are read as
    /// UTF-8.
    ///
    /// Token strings are literal text. The vocab holds Tamil clusters learned
    /// from raw text, so a token like `café` is the word and not three bytes,
    /// and decoding it as bytes would replace it with a replacement character.
    fn decode_inner(&self, token_ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(token_ids.len() * 4);
        for &id in token_ids {
            let Some(token) = self.id_to_token.get(id as usize) else {
                continue;
            };
            match parse_byte_token(token) {
                Some(b) => bytes.push(b),
                None => bytes.extend_from_slice(token.as_bytes()),
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
