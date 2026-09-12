use crate::pretokenizer::TamilPretokenizer;
use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

// ============================================================================
// TYPE ALIAS AND HELPER FUNCTIONS FOR PACKED PAIR STORAGE
// ============================================================================

type TokId = u32;

#[inline]
fn pack_pair(a: TokId, b: TokId) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline]
fn unpack_pair(p: u64) -> (TokId, TokId) {
    ((p >> 32) as TokId, (p & 0xFFFF_FFFF) as TokId)
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeapEntry {
    key: u64,
    count: u32,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.count.cmp(&other.count) {
            Ordering::Equal => other.key.cmp(&self.key), // Reversed like HuggingFace
            other => other,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
struct WordData {
    tokens: Vec<TokId>,
    freq: u32,
}

#[derive(Serialize, Deserialize)]
struct TokenizerData {
    vocab: HashMap<String, u32>,
    merges: Vec<Vec<String>>,
    metadata: TokenizerMetadata,
}

#[derive(Serialize, Deserialize)]
struct TokenizerMetadata {
    vocab_size: usize,
    num_merges: usize,
    training_time_seconds: f64,
    unique_words_processed: usize,
}

// ============================================================================
// MAIN TRAINER STRUCT
// ============================================================================

#[allow(dead_code)]
pub struct OptimizedBPETrainer {
    pretokenizer: TamilPretokenizer,

    // Token management
    token_to_id: HashMap<String, TokId>,
    id_to_token: Vec<String>,
    next_token_id: TokId,

    // Word storage
    words: Vec<WordData>,

    // Pair frequency: pack_pair(a, b) -> frequency
    pair_freq: HashMap<u64, u32>,

    // Position tracking: pack_pair(a, b) -> set of word IDs
    pair_positions: HashMap<u64, HashSet<usize>>,

    // Priority queue
    pair_heap: BinaryHeap<HeapEntry>,

    // Merge history
    merges: Vec<(String, String)>,
    tamil_clusters: HashSet<String>,
    total_time: Duration,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl OptimizedBPETrainer {
    pub fn new(initial_vocab: HashMap<String, u32>) -> Self {
        // rayon picks its own pool size; overriding it here would ignore the
        // user's RAYON_NUM_THREADS and pin every trainer to one machine's cores.
        let tamil_clusters: HashSet<String> = initial_vocab.keys().cloned().collect();
        let vocab_len = initial_vocab.len() as TokId;

        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        // Sort lexicographically for deterministic ordering
        let mut sorted_vocab: Vec<_> = initial_vocab.into_iter().collect();
        sorted_vocab.sort_by(|a, b| a.0.cmp(&b.0));

        for (idx, (token, _)) in sorted_vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), idx as TokId);
            id_to_token.push(token.clone());
        }

        Self {
            pretokenizer: TamilPretokenizer::new(),
            token_to_id,
            id_to_token,
            next_token_id: vocab_len,
            words: Vec::new(),
            pair_freq: HashMap::new(),
            pair_positions: HashMap::new(),
            pair_heap: BinaryHeap::new(),
            merges: Vec::new(),
            tamil_clusters,
            total_time: Duration::ZERO,
        }
    }

    pub fn initialize_from_corpus(&mut self, corpus_path: &str) -> std::io::Result<()> {
        println!("Reading and tokenizing corpus in parallel...");
        let start_time = Instant::now();

        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(corpus_path)?;
        let reader = BufReader::with_capacity(16 * 1024 * 1024, file);

        // Collect directly into io::Result: dropping unreadable lines here would
        // silently train on a truncated corpus.
        let all_lines: Vec<String> = reader.lines().collect::<std::io::Result<_>>()?;

        println!("Processing {} lines in parallel...", all_lines.len());

        let chunk_size = all_lines.len() / rayon::current_num_threads().max(1);
        let chunk_size = chunk_size.max(50000);

        let partial_results: Vec<HashMap<Vec<String>, u32>> = all_lines
            .par_chunks(chunk_size)
            .map(|chunk| {
                let pretok = TamilPretokenizer::new();
                let mut local_freqs = HashMap::new();

                for line in chunk {
                    let chunks = pretok.pretokenize(line);
                    for chunk_text in &chunks {
                        let tokens = self.segment_into_clusters(chunk_text);
                        *local_freqs.entry(tokens).or_insert(0) += 1;
                    }
                }
                local_freqs
            })
            .collect();

        let mut word_freqs_temp = HashMap::new();
        for local_map in partial_results {
            for (tokens, freq) in local_map {
                *word_freqs_temp.entry(tokens).or_insert(0) += freq;
            }
        }

        println!(
            "Building pair statistics for {} unique words...",
            word_freqs_temp.len()
        );

        for (tokens, freq) in word_freqs_temp {
            let token_ids = self.convert_to_ids(&tokens);
            let word_id = self.words.len();

            self.words.push(WordData {
                tokens: token_ids.clone(),
                freq,
            });

            self.update_pairs_for_word(word_id, &token_ids, freq, true);
        }

        self.rebuild_heap();

        let elapsed = start_time.elapsed();
        println!(
            "Cached {} unique words from {} lines in {:.2}s\n",
            self.words.len(),
            all_lines.len(),
            elapsed.as_secs_f64()
        );

        Ok(())
    }

    fn segment_into_clusters(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut matched = false;

            if i + 2 < chars.len() {
                let cluster: String = chars[i..i + 3].iter().collect();
                if self.tamil_clusters.contains(&cluster) {
                    tokens.push(cluster);
                    i += 3;
                    matched = true;
                }
            }

            if !matched && i + 1 < chars.len() {
                let cluster: String = chars[i..i + 2].iter().collect();
                if self.tamil_clusters.contains(&cluster) {
                    tokens.push(cluster);
                    i += 2;
                    matched = true;
                }
            }

            if !matched {
                tokens.push(chars[i].to_string());
                i += 1;
            }
        }
        tokens
    }

    fn convert_to_ids(&mut self, tokens: &[String]) -> Vec<TokId> {
        tokens
            .iter()
            .map(|token| {
                if let Some(&id) = self.token_to_id.get(token) {
                    id
                } else {
                    let id = self.next_token_id;
                    self.next_token_id += 1;
                    self.token_to_id.insert(token.clone(), id);
                    self.id_to_token.push(token.clone());
                    id
                }
            })
            .collect()
    }

    fn update_pairs_for_word(&mut self, word_id: usize, token_ids: &[TokId], freq: u32, add: bool) {
        for i in 0..token_ids.len().saturating_sub(1) {
            let pair_key = pack_pair(token_ids[i], token_ids[i + 1]);

            if add {
                *self.pair_freq.entry(pair_key).or_insert(0) += freq;
                self.pair_positions
                    .entry(pair_key)
                    .or_default()
                    .insert(word_id);
            } else {
                if let Some(count) = self.pair_freq.get_mut(&pair_key) {
                    *count = count.saturating_sub(freq);
                    if *count == 0 {
                        self.pair_freq.remove(&pair_key);
                    }
                }
                if let Some(positions) = self.pair_positions.get_mut(&pair_key) {
                    positions.remove(&word_id);
                }
            }
        }
    }

    fn rebuild_heap(&mut self) {
        self.pair_heap.clear();

        // Collect and sort by (freq DESC, key ASC) for deterministic ordering
        let mut pairs_vec: Vec<(u64, u32)> = self
            .pair_freq
            .iter()
            .filter(|&(_, &freq)| freq >= 1)
            .map(|(&key, &freq)| (key, freq))
            .collect();

        // Sort by frequency desc, then by packed key asc (lexicographic)
        pairs_vec.sort_by(|a, b| {
            match b.1.cmp(&a.1) {
                // Frequency descending
                Ordering::Equal => a.0.cmp(&b.0), // Key ascending (lexicographic)
                other => other,
            }
        });

        for (pair_key, freq) in pairs_vec {
            self.pair_heap.push(HeapEntry {
                key: pair_key,
                count: freq,
            });
        }
    }

    pub fn train(&mut self, num_merges: usize) {
        println!("Starting BPE training with {} merges...", num_merges);
        let train_start = Instant::now();

        for merge_iteration in 0..num_merges {
            let merge_start = Instant::now();

            // Lazy stale validation
            let best_pair_opt = loop {
                if let Some(entry) = self.pair_heap.pop() {
                    if let Some(&current_freq) = self.pair_freq.get(&entry.key) {
                        if current_freq == entry.count {
                            break Some(entry);
                        }
                    }
                } else {
                    break None;
                }
            };

            if let Some(entry) = best_pair_opt {
                let (id_a, id_b) = unpack_pair(entry.key);
                let token1 = self.id_to_token[id_a as usize].clone();
                let token2 = self.id_to_token[id_b as usize].clone();
                let merged = format!("{}{}", token1, token2);

                let new_id = self.next_token_id;
                self.next_token_id += 1;
                self.token_to_id.insert(merged.clone(), new_id);
                self.id_to_token.push(merged.clone());
                self.merges.push((token1.clone(), token2.clone()));

                let affected_words: Vec<usize> = self
                    .pair_positions
                    .get(&entry.key)
                    .map(|set| set.iter().copied().collect())
                    .unwrap_or_default();

                let mut new_pairs_to_add: HashMap<u64, u32> = HashMap::new();

                for word_id in affected_words {
                    let word = &self.words[word_id];
                    let old_tokens = word.tokens.clone();
                    let word_freq = word.freq;

                    self.update_pairs_for_word(word_id, &old_tokens, word_freq, false);

                    let mut new_tokens = Vec::with_capacity(old_tokens.len());
                    let mut i = 0;
                    while i < old_tokens.len() {
                        if i < old_tokens.len() - 1
                            && old_tokens[i] == id_a
                            && old_tokens[i + 1] == id_b
                        {
                            new_tokens.push(new_id);
                            i += 2;
                        } else {
                            new_tokens.push(old_tokens[i]);
                            i += 1;
                        }
                    }

                    self.words[word_id].tokens = new_tokens.clone();
                    self.update_pairs_for_word(word_id, &new_tokens, word_freq, true);

                    // Collect ONLY new pairs involving new_id (for heap)
                    for i in 0..new_tokens.len().saturating_sub(1) {
                        let pair_key = pack_pair(new_tokens[i], new_tokens[i + 1]);
                        let (a, b) = unpack_pair(pair_key);

                        if a == new_id || b == new_id {
                            *new_pairs_to_add.entry(pair_key).or_insert(0) += word_freq;
                        }
                    }
                }

                // Add collected pairs to heap (use actual frequency from pair_freq, not accumulated)
                let mut new_pairs_vec: Vec<(u64, u32)> = new_pairs_to_add
                    .into_iter()
                    .filter_map(|(pair_key, _)| {
                        self.pair_freq.get(&pair_key).and_then(|&freq| {
                            if freq >= 1 {
                                Some((pair_key, freq))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

                // Sort for deterministic heap insertion
                new_pairs_vec.sort_by(|a, b| match b.1.cmp(&a.1) {
                    Ordering::Equal => a.0.cmp(&b.0),
                    other => other,
                });

                for (pair_key, freq) in new_pairs_vec {
                    self.pair_heap.push(HeapEntry {
                        key: pair_key,
                        count: freq,
                    });
                }

                let merge_elapsed = merge_start.elapsed();
                if merge_iteration % 100 == 0 {
                    println!(
                        "  Iter {}: '{}' + '{}' → '{}' (freq={}) | {:.3}s",
                        merge_iteration,
                        token1,
                        token2,
                        merged,
                        entry.count,
                        merge_elapsed.as_secs_f64()
                    );
                }
            } else {
                println!("No more pairs to merge at iteration {}", merge_iteration);
                break;
            }
        }

        self.total_time = train_start.elapsed();
        println!(
            "\nTraining complete in {:.2}s ({:.2}m)",
            self.total_time.as_secs_f64(),
            self.total_time.as_secs_f64() / 60.0
        );
    }

    pub fn get_final_vocab(&self) -> HashMap<String, u32> {
        let mut vocab = HashMap::new();
        for (idx, token) in self.id_to_token.iter().enumerate() {
            vocab.insert(token.clone(), idx as u32);
        }
        vocab
    }

    // Add special tokens + reserved tokens to vocab
    pub fn save_tokenizer_with_reserved_tokens(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut vocab_map: HashMap<String, u32> = HashMap::new();
        for (idx, token) in self.id_to_token.iter().enumerate() {
            vocab_map.insert(token.clone(), idx as u32);
        }

        let base_vocab_size = vocab_map.len() as u32;

        // Named special tokens
        let special_tokens = vec![
            "<|startoftext|>",
            "<|endoftext|>",
            "<|pad|>",
            "<|unk|>",
            "<|mask|>",
            "<|question|>",
            "<|answer|>",
            "<|code|>",
            // "<|tamil|>",
            // "<|english|>",
        ];

        let mut current_id = base_vocab_size;

        // Add named tokens
        for special_token in &special_tokens {
            vocab_map.insert(special_token.to_string(), current_id);
            current_id += 1;
        }

        let named_tokens_end = current_id;

        // Add reserved tokens (100 slots)
        let reserved_count = 100u32;
        for i in 0..reserved_count {
            let reserved_token = format!("<|reserved_{}|>", current_id + i);
            vocab_map.insert(reserved_token, current_id + i);
        }

        current_id += reserved_count;

        let merges_list: Vec<Vec<String>> = self
            .merges
            .iter()
            .map(|(t1, t2)| vec![t1.clone(), t2.clone()])
            .collect();

        let tokenizer_data = TokenizerData {
            vocab: vocab_map,
            merges: merges_list,
            metadata: TokenizerMetadata {
                vocab_size: current_id as usize,
                num_merges: self.merges.len(),
                training_time_seconds: self.total_time.as_secs_f64(),
                unique_words_processed: self.words.len(),
            },
        };

        let json_string = serde_json::to_string_pretty(&tokenizer_data)?;
        let mut file = File::create(path)?;
        file.write_all(json_string.as_bytes())?;

        println!(
            "Saved tokenizer with special + reserved tokens to: {}",
            path
        );
        println!("   Base vocab: 0-{}", base_vocab_size - 1);
        println!(
            "   Named special tokens: {}-{}",
            base_vocab_size,
            named_tokens_end - 1
        );
        println!(
            "   Reserved tokens: {}-{}",
            named_tokens_end,
            current_id - 1
        );
        println!("   Total vocab size: {}", current_id);

        Ok(())
    }

    pub fn save_vocab_json(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;

        let mut vocab_map = HashMap::new();
        for (idx, token) in self.id_to_token.iter().enumerate() {
            vocab_map.insert(token.clone(), idx as u32);
        }

        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &vocab_map)?;
        println!("Saved vocabulary to {}", path);
        Ok(())
    }

    pub fn save_merges_json(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;

        #[derive(Serialize)]
        struct MergesData {
            merges: Vec<Vec<String>>,
            count: usize,
        }

        let merges_list: Vec<Vec<String>> = self
            .merges
            .iter()
            .map(|(t1, t2)| vec![t1.clone(), t2.clone()])
            .collect();

        let merges_data = MergesData {
            merges: merges_list,
            count: self.merges.len(),
        };

        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &merges_data)?;
        println!("Saved merges to {}", path);
        Ok(())
    }

    pub fn print_stats(&self) {
        println!("\n=== Training Statistics ===");
        println!("Total vocabulary size: {}", self.id_to_token.len());
        println!("Number of merge rules: {}", self.merges.len());
        println!("Unique words: {}", self.words.len());
        println!(
            "Total training time: {:.2}s ({:.2}m)",
            self.total_time.as_secs_f64(),
            self.total_time.as_secs_f64() / 60.0
        );
    }
}
