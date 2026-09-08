# Tamil Tokenizer

A high-performance **Byte Pair Encoding (BPE)** tokenizer library for **Tamil text**, written in Rust. Designed for training and using custom tokenizers on Tamil corpora with an Aho-Corasick-based fast encoder.

## Features

- **Tamil-aware vocabulary** — 247 pre-built Tamil character clusters (consonant+vowel sign combinations) as base tokens
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
├── tests/lib.rs            # Test suite (37 tests)
├── scripts/evaluate.py     # Compression and fertility report (needs no extra deps)
├── scripts/extract_wiki_*.py # One-off Wikipedia corpus extraction
└── data/tamil_corpus.txt   # 117-line sample corpus
```

## Quickstart

### Pre-trained Model

A 32K-vocab tokenizer trained on Tamil Wikipedia is published at
**[barryallen16/tamil-tokenizer-wiki](https://huggingface.co/barryallen16/tamil-tokenizer-wiki)**.
It is a separate, larger artifact from the 1,015-token model in `output/` produced by the
commands below.

```bash
# Download the tokenizer JSON directly
curl -L -o tamil_tokenizer_wiki.json \
  https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main/tamil_tokenizer_wiki.json

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
let vocab = tokenizer.get_sorted_vocab();

// Create trainer and load corpus
let mut trainer = OptimizedBPETrainer::new(vocab);
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
| Trained merges | 297 |
| Corpus compression | 2.56 bytes/token |
| Tamil tokens/word | 3.64 |
| Encode throughput | 3-12 MB/s |
| Time per document | 4-16 us |

Throughput varies run to run; the corpus is only 5 KB, so a few hundred documents fit
in cache and the measurement is dominated by warmup.

### Compression by script

Measured with `bench_encode` on a 20 KB slice of each input:

| Input | Bytes/token |
|-------|------------:|
| Tamil | 2.66 |
| English | 1.87 |
| Code | 1.32 |

English and code compress less because the training corpus is Tamil-only, so few
ASCII merges were learned.

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
them (907-914 named, 915-1014 reserved here), and the encoder reuses those IDs
instead of remapping them.

### The published 32K model

The numbers reported on the Hugging Face model card for `barryallen16/tamil-tokenizer-wiki`
(32,015 vocab, 29,999 merges, 2.57 bytes/token, 10.95 MB/s) refer to that separate
artifact. They are not measurements of the code in this repository, and no script here
reproduces them.

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

The 32K published model is documented on its own model card:
[barryallen16/tamil-tokenizer-wiki](https://huggingface.co/barryallen16/tamil-tokenizer-wiki).

Note that `encode`, `decode`, and `main` read the file's `vocab` key. They use
`FastBPEEncoder`, which applies the `merges` field in rank order. Token IDs may
still differ from the model card's examples (different pretokenizer and base vocab).

### Usage with the Rust CLI

```bash
# Download the tokenizer
curl -L -o tamil_tokenizer_wiki.json \
  https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main/tamil_tokenizer_wiki.json

# Encode text
cargo run --release --bin encode -- tamil_tokenizer_wiki.json "வணக்கம்!"
```

---

## License

MIT
