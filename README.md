# Tamil Tokenizer

A high-performance **Byte Pair Encoding (BPE)** tokenizer library for **Tamil text**, written in Rust. Designed for training and using custom tokenizers on Tamil corpora with an Aho-Corasick-based fast encoder.

## Features

- **Tamil-aware base vocabulary** — 524 tokens with no training: 256 byte tokens, 216 consonant+vowel-sign clusters, 31 standalone Tamil characters, and 21 digits and symbols
- **Optimized BPE training** — Parallel corpus processing via [rayon](https://docs.rs/rayon), packed pair storage, and a priority queue for fast merge selection
- **Fast encoding** — Aho-Corasick automaton for O(n) multi-pattern matching during tokenization
- **Special token support** — BOS/EOS, language tags, question/answer markers, and 100 reserved slots (o200k-style layout)
- **GPT-2 compatible byte mappings** — Control characters mapped to printable Unicode for clean token display

## Project Structure

```
.
├── lib.rs                  # Library root — re-exports public types
├── vocab.rs                # TamilTokenizer — base vocabulary + byte mappings
├── pretokenizer.rs         # TamilPretokenizer — regex-based text splitting
├── trainer.rs              # OptimizedBPETrainer — parallel BPE training
├── encoder.rs              # FastBPEEncoder — encode/decode with Aho-Corasick
├── train.rs                # CLI: train a tokenizer from a corpus file
├── encode.rs               # CLI: encode text to token IDs
├── decode.rs               # CLI: decode token IDs back to text
├── benchmark.rs            # Corpus benchmark (Rust)
├── bench_encode.rs         # CLI: encode one text or stdin lines (for scripts)
├── bench_speed.rs          # CLI: JSON throughput report (for scripts)
├── test_special_tokens.rs  # Special token integration tests
├── main.rs                 # Interactive encoder/decoder REPL
├── tests/lib.rs            # Test suite (43 tests)
├── scripts/evaluate.py     # Compression and fertility report (needs no extra deps)
├── scripts/extract_wiki_*.py # One-off Wikipedia corpus extraction
└── data/tamil_corpus.txt   # 117-line sample corpus
```

## Quickstart

### Pre-trained Model

The Hugging Face repo
**[barryallen16/tamil-tokenizer-wiki](https://huggingface.co/barryallen16/tamil-tokenizer-wiki)**
publishes a trained 32K-merge tokenizer alongside the training corpora.

```bash
# Download the published 32K tokenizer (2.7 MB)
curl -L -o output/tamil_tokenizer_wiki.json \
  https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main/tamil_tokenizer_wiki.json

# It loads with this repo's encoder:
cargo run --release --bin encode -- ./output/tamil_tokenizer_wiki.json "வணக்கம்"
# 490 334 2331

# Download the 38 MB training corpus
curl -L -o data/tamil_wiki_corpus.txt \
  https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main/tamil_wiki_corpus.txt
```

`scripts/evaluate.py` accepts the tokenizer path as an argument and defaults to the
local `output/tamil_tokenizer.json`.

### 1. Train a Tokenizer

```bash
# Build the training binary
cargo build --release

# Train on a Tamil corpus with 50,000 merge operations
cargo run --release --bin train -- ./data/tamil_corpus.txt 50000 ./output/tamil_tokenizer.json
```

This produces a JSON vocab file at `output/tamil_tokenizer.json`.

### 2. Encode Text

```bash
# Encode a string to token IDs
cargo run --release --bin encode -- ./output/tamil_tokenizer.json "வணக்கம்! இது தமிழ்"
```

### 3. Decode Token IDs

```bash
# Decode token IDs back to text
cargo run --release --bin decode -- ./output/tamil_tokenizer.json 224 174 181
# வ
```

`main` accepts an optional vocab path, defaulting to `output/tamil_tokenizer.json`:
`cargo run --release --bin main -- ./output/tamil_tokenizer.json`

### 4. Interactive Mode

```bash
cargo run --release --bin main
# Enter text to encode/decode interactively
```

### 5. Run Benchmarks

```bash
# Rust encoder benchmark
cargo run --release --bin benchmark

# Compression and fertility report (no extra dependencies)
python scripts/evaluate.py
python scripts/evaluate.py ./output/tamil_tokenizer.json
```

## Library Usage (Rust)

### Training

```rust
use tamil_tokenizer::{TamilTokenizer, OptimizedBPETrainer};

// Initialize base vocabulary
let tokenizer = TamilTokenizer::new();

// Create trainer and load corpus
let mut trainer = OptimizedBPETrainer::new(tokenizer.vocab);
trainer.initialize_from_corpus("./data/tamil_corpus.txt");

// Train with 50,000 merges
trainer.train(50000);

// Save trained tokenizer
trainer.save_tokenizer_with_reserved_tokens("./output/tamil_tokenizer.json")
    .expect("Failed to save");
```

### Encoding & Decoding

```rust
use tamil_tokenizer::FastBPEEncoder;
use rustc_hash::FxHashMap;
use std::fs;

// Load trained vocabulary. The file is {vocab, merges, metadata}, so it must be
// unwrapped -- deserializing the whole file into a flat map fails with
// "invalid type: map, expected u32".
let data = fs::read_to_string("output/tamil_tokenizer.json").unwrap();
let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
let vocab: FxHashMap<String, u32> = parsed["vocab"]
    .as_object()
    .unwrap()
    .iter()
    .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
    .collect();

let encoder = FastBPEEncoder::new(vocab).unwrap();

// Encode with automatic BOS/EOS
let token_ids = encoder.encode("வணக்கம்! இது தமிழ் tokenizer.");
println!("Tokens: {:?}", token_ids);

// Encode without special tokens
let token_ids = encoder.encode_ordinary("வணக்கம்!");

// Decode back to text
let text = encoder.decode(&token_ids);
println!("Decoded: {}", text);

// Decode skipping special tokens
let clean = encoder.decode_skip_special(&token_ids);
```

### Special Tokens

The encoder supports an o200k-style token layout:

| Range | Tokens |
|-------|--------|
| `0` – `base_vocab_size` | Base vocabulary (Tamil clusters, bytes, etc.) |
| `base` – `base + 8` | Named specials: `<\|startoftext\|>`, `<\|endoftext\|>`, `<\|pad\|>`, `<\|unk\|>`, `<\|mask\|>`, `<\|question\|>`, `<\|answer\|>`, `<\|code\|>` |
| `base + 9` – `base + 109` | 100 reserved tokens: `<\|reserved_0\|>` … `<\|reserved_99\|>` |

```rust
// Check if a token ID is special
let is_special = encoder.is_special_token(token_id);

// Get ID of a named special token
let bos_id = encoder.get_special_token_id("<|startoftext|>");
```

## Architecture

```
Text Input
    │
    ▼
┌──────────────────┐
│  Pretokenizer    │  Regex-based splitting (Tamil-aware Unicode patterns)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Aho-Corasick    │  O(n) multi-pattern matching against vocabulary
│  Automaton       │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Token IDs       │  Output vector of u32 token IDs
└──────────────────┘
```

The BPE trainer uses a **max-heap** to efficiently select the most frequent token pair at each merge step, with packed `u64` pair keys for cache-friendly storage and `rayon` parallelism during corpus loading.

## Benchmarks

All numbers below were produced by `cargo run --release --bin benchmark` against
`output/tamil_tokenizer.json` on this repo's `data/tamil_corpus.txt`
(117 lines, 5,161 bytes, 424 unique words). They are reproducible from a clean checkout.

### Measured: local 1,015-token model

| Metric | Value |
|--------|------:|
| Vocabulary size | 1,015 (907 base + 8 named specials + 100 reserved) |

The 907 base is larger than the 524-token untrained base because the trainer
also mints IDs for Tamil clusters it finds in the corpus that are not already in
the base vocab; that count depends on the corpus.
| Trained merges | 297 (1,159 with the frequency floor relaxed; see below) |
| Corpus compression | 2.56 bytes/token |
| Tamil tokens/word | 3.64 |
| Encode throughput (10K wiki lines) | ~23 MB/s (1K vocab), ~15 MB/s (30K wiki vocab) |

Throughput measured with `bench_speed` over 10,000 wiki lines x 3 repeats
(3.1 MB per pass); run-to-run variance is ~1%. The 30K vocab is slower per byte
because its automaton and rank scans are larger, but it emits half the tokens.

### Head-to-head, same inputs, measured here

Each tokenizer encoded identical text: the 117-line Tamil corpus plus fixed
English and code samples. Opponents were trained fresh on the same corpus where
applicable (HF BPE at 1,000 vocab, SentencePiece BPE at 500; ours is 1,015).
Reproduce with `uv pip install tiktoken sentencepiece tokenizers`, then
`python scripts/compare.py` (missing libraries are skipped). No speed claims:
timing a 5 KB corpus measures warmup, not throughput.

| Input | Ours | tiktoken cl100k | HF BPE 1K | SPM BPE 500 |
|-------|-----:|----------------:|----------:|------------:|
| Tamil B/tok | 2.77 | 2.38 | 3.98 | 2.71 |
| Tamil tok/word | 3.64 | 4.24 | 2.53 | 3.72 |
| English B/tok | 1.71 | 5.56 | 2.05 | 1.78 |
| Code B/tok | 1.41 | 2.76 | 1.73 | 1.41 |

In multipliers, on Tamil we use 1.16x fewer tokens than tiktoken; HF's BPE uses
1.44x fewer than us on the same training data. On English, tiktoken's 100K
English-centric vocab uses 3.3x fewer tokens than us, as expected from a
Tamil-only 1K vocab. SentencePiece ties us almost exactly.

(The 2.77 here vs 2.56 in the table above is line handling: `benchmark` trims
lines and scores the first 100 docs, while the comparison scores all 117 raw
lines identically for every tokenizer.)

Ablation: capping HF at our 297 merges gives 1,873 tokens vs our 1,865 --
identical. The gap above is merge budget (793 vs 297), not algorithm: HF ran
with `min_frequency=1` while our trainer required pair frequency >= 2.
Relaxing our floor to >= 1 learns 1,159 merges on this corpus and reaches 5.15
bytes/token, ahead of HF's 3.98.

### Wiki-scale training

Training on `data/tamil_wiki_corpus.txt` (127,889 lines, 38.8 MB, gitignored)
scales cleanly. Compression on held-out wiki lines by merge budget:

| Merges | Train time | Wiki B/tok |
|-------:|-----------:|-----------:|
| 1,000 | seconds | 5.79 |
| 5,000 | 1.1 s | 7.07 |
| 30,000 | 1.4 s | 8.31 |

Even at 30K merges the tail pair frequency is 11, so no singleton junk appears
and late merges are legitimate words (`தலைநகராக`, `பங்காற்றினார்`). A 5,000-line
wiki fuzz through encode/decode round-trips byte-exact on both the 1K and 30K
vocabs. The small corpus still favors its in-domain model (2.77 vs 2.71 for the
30K wiki model): domain beats budget. Note the wiki corpus is noisy -- about a quarter of the
1K model's merges are ASCII markup fragments (`colo+r`, `j+pg`) -- so a
production run wants markup stripping in extraction first.

### How encoding works

The trainer learns merges over raw text, so Tamil clusters and merged tokens are
stored as raw Tamil strings. `FastBPEEncoder` encodes each pretokenized chunk in
three steps: longest-match over base tokens only (merge results excluded), then
standard BPE -- repeatedly merging the adjacent pair with the lowest merge rank
until no ranked pair remains -- then one token ID per symbol. Anything without an
ID falls back to one byte-level token per UTF-8 byte, which keeps encoding
lossless for any input.

Byte tokens use GPT-2 style stand-in characters (U+0100 to U+01A1). Tokens containing
a stand-in are kept out of the matcher, so real text containing those characters
(for example Polish or Turkish letters) is never mistaken for raw bytes.

Special tokens keep the IDs stored in the file: a trained vocab already contains
them, and the encoder reuses those IDs instead of remapping them. The exact
where they start depends on the corpus, because the trainer mints IDs for
clusters it discovers while reading.

### The published 32K model

The published model is 32,142 vocab entries (IDs 0–32141, contiguous) and 30,000
ordered merges: 32,034 content tokens, 8 named specials, and 100 reserved slots.
It was trained on the 38.8 MB wiki corpus in about 1.4 s.

Two caveats on using it from Python: the file is in this repo's
`{vocab, merges, metadata}` format, so `tokenizers.Tokenizer.from_file` will not
load it — use the Rust CLI or `lib.rs` directly. And this repo makes no
throughput comparison against other tokenizer libraries; the figures above are
its own measurements only.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `regex` | Unicode-aware pretokenization patterns |
| `rayon` | Parallel corpus processing |
| `serde` / `serde_json` | Vocab serialization |
| `rustc_hash` | Fast hash maps (FxHashMap) |
| `ahash` | Hash maps for trainer internals |
| `aho_corasick` | Multi-pattern matching encoder |

## Hugging Face Integration

The trained tokenizer is published at
[barryallen16/tamil-tokenizer-wiki](https://huggingface.co/barryallen16/tamil-tokenizer-wiki)
as `tamil_tokenizer_wiki.json`, along with the corpora used to train it.

### Usage with the Rust CLI

```bash
# Encode text with the published tokenizer
cargo run --release --bin encode -- ./output/tamil_tokenizer_wiki.json "வணக்கம்!"
```
