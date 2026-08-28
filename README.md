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
├── benchmark_tiktoken.py   # Comparison benchmark (Python tiktoken)
├── test_special_tokens.rs  # Special token integration tests
└── main.rs                 # Interactive encoder/decoder REPL
```

## Quickstart

### Pre-trained Model

> 🔗 **[Hugging Face: tamil-tokenizer-wiki](https://huggingface.co/barryallen16/tamil-tokenizer-wiki)**
>
> A pre-trained tokenizer with a 32K vocab, trained on 128K lines of Tamil Wikipedia text.
>
> ```bash
> # Download the tokenizer JSON directly
> curl -L -o tamil_tokenizer_wiki.json \
>   https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main/tamil_tokenizer_wiki.json
> ```

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
cargo run --release --bin decode -- ./output/tamil_tokenizer.json 245 1023 567
```

### 4. Interactive Mode

```bash
cargo run --release --bin main
# Enter text to encode/decode interactively
```

### 5. Run Benchmarks

```bash
# Rust encoder benchmark
cargo run --release --bin benchmark

# Compare with tiktoken (Python)
pip install tiktoken
python benchmark_tiktoken.py
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

// Load trained vocabulary
let data = fs::read_to_string("output/tamil_tokenizer.json").unwrap();
let vocab: FxHashMap<String, u32> = serde_json::from_str(&data).unwrap();

let encoder = FastBPEEncoder::new(vocab);

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

Compared against **tiktoken** (OpenAI), **SentencePiece** (Google), and **HuggingFace Tokenizers** on Tamil, English, and mixed text.

### Speed Comparison

| Tokenizer | Vocab | Tamil B/Tok | Tamil Speed | English B/Tok | English Speed |
|-----------|------:|------------:|------------:|--------------:|--------------:|
| **Ours (Rust BPE)** | **32K** | **13.44** | **52.9 MB/s** | **2.21** | **17.8 MB/s** |
| SentencePiece (BPE) | 30K | 17.93 | 4.7 MB/s | 3.28 | 2.5 MB/s |
| HF Tokenizers (BPE) | 30K | 18.62 | 4.3 MB/s | 3.50 | 1.3 MB/s |
| tiktoken (cl100k) | 100K | 2.06 | 3.2 MB/s | 5.72 | 4.1 MB/s |

### Evaluation Metrics

| Language | Bytes/Token | Tokens/Word | Interpretation |
|----------|------------:|------------:|----------------|
| **Tamil** | 5.85 | 4.22 | Agglutinative — more tokens per word |
| **English** | 6.71 | 1.03 | Words encode efficiently |
| **Code** | 3.43 | 1.66 | Symbol-heavy — lower compression |

- **Fertility**: Tamil averages 4.22 tokens per word vs English 1.03
- **Compression**: Code has the lowest bytes/token (3.43) due to symbol density

> **Bottom line**: Our tokenizer is **11× faster** than SentencePiece and **16× faster** than HF Tokenizers on Tamil text.

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

This tokenizer is published on 🤗 Hugging Face Hub:

- **Model card:** [tamil-tokenizer-wiki](https://huggingface.co/barryallen16/tamil-tokenizer-wiki)
- **Tokenizer JSON:** [`tamil_tokenizer_wiki.json`](https://huggingface.co/barryallen16/tamil-tokenizer-wiki/blob/main/tamil_tokenizer_wiki.json)

### Usage with Hugging Face Tokenizers library

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("barryallen16/tamil-tokenizer-wiki")
encoded = tokenizer("வணக்கம்! இது தமிழ் tokenizer.")
print(encoded["input_ids"])
```

### Usage with Rust (this crate)

```bash
# Download the tokenizer
wget https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main/tamil_tokenizer_wiki.json

# Encode text
cargo run --release --bin encode -- tamil_tokenizer_wiki.json "வணக்கம்!"
```

---

## License

MIT
