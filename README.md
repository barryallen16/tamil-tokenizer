# Tamil Tokenizer

A BPE tokenizer library for Tamil text, written in Rust.
It trains on custom Tamil corpora, encodes quickly, and exports to Hugging Face without changing token IDs.
On the same 20M-token pretraining budget it is 10.1-10.2% better than `indictok-256k` and 36.7-36.9% better than mBERT in bits per byte on Tamil wiki text, and it is 16.5 points more accurate on six-class Tamil news classification.

## Features

- Tamil-aware base vocabulary of 1,135 tokens with no training: 256 byte tokens, 12 independent vowels, 23 standalone consonants, 276 consonant+vowel-sign clusters (including Grantha `ஜ ஶ ஷ ஸ ஹ`), 4 special characters (`ஃ ஂ ௐ ௗ`), 15 compound ligatures (`க்ஷ`, `ஸ்ரீ`), 528 conjunct stems, 10 Tamil digits, and 11 extra characters.
- The conjunct stems are what moved compression.
  Tamil joins consonants constantly (`க்ஷ`, `ஸ்ரீ`, `க்ட`, `ண்ட`), and each one used to cost 3-4 tokens.
- Training runs in parallel through [rayon](https://docs.rs/rayon) and uses packed pair storage plus a priority queue for merge selection.
- Encoding uses a packed merge table with a per-pair rule cache, so a pair is looked up once per chunk instead of once per merge.
- `encode_batch` and `encode_ordinary_batch` split a document set across all cores.
- `export_hf` writes `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, and `merges.txt`.
  The output is checked against Python `tokenizers` and `transformers`.
- The Rust encoder and the Hugging Face export read the same vocabulary and merge list, so they emit the same token IDs for the same text.
- Special tokens cover BOS/EOS, language tags, question and answer markers, and 100 reserved slots in an o200k-style layout.
- Byte mapping uses the exact GPT-2 byte-to-unicode table, so the export matches Hugging Face's `ByteLevel`.

## Training datasets

The tokenizer trains on a balanced mixture of three complementary corpora:

- **Tamil Wikipedia**: Formal encyclopedia text providing core orthographic coverage and domain terminology.
- **IndicCorp v2**: Multi-source Tamil news and web crawl articles providing modern written prose.
- **YouTube Video Transcripts**: 1,400 conversational Tamil video transcripts (896,641 lines, 71 MB) sourced from `sivabharani-comments`.
  This adds spoken Tamil patterns and colloquial vocabulary absent from written publications.

Corpus preparation and mixing are handled by `scripts/prepare_corpus.py`.

## Quickstart

### Python

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("barryallen16/tamil-tokenizer-wiki")
tok.encode("வணக்கம் இது தமிழ்").input_ids
```

The 120K model is at the repository root; the 30K model is in `tamil_tokenizer_wiki_hf/`:

```python
tok_30k = AutoTokenizer.from_pretrained(
    "barryallen16/tamil-tokenizer-wiki", subfolder="tamil_tokenizer_wiki_hf"
)
```

### CLI

The unified binary supports training, encoding, decoding, evaluation, and export:

```bash
# Train on a corpus with a target merge count
cargo run --release -- train ./data/tamil_corpus.txt 50000 ./output/tamil_tokenizer.json

# Encode text to token IDs
cargo run --release -- encode ./output/tamil_tokenizer.json "வணக்கம்!"

# Decode token IDs back to text
cargo run --release -- decode ./output/tamil_tokenizer.json 224 174 181

# Interactive REPL
cargo run --release -- repl ./output/tamil_tokenizer.json

# Score compression and throughput in-process
cargo run --release -- score ./output/tamil_tokenizer.json ./data/tamil_corpus.txt sample

# Export Hugging Face artifacts
cargo run --release -- export ./output/tamil_tokenizer.json ./output/hf_export
```

### Rust library API

```rust
use tamil_tokenizer::{load_tokenizer, FastBPEEncoder, OptimizedBPETrainer, TamilTokenizer};

// Train
let base = TamilTokenizer::new();
let mut trainer = OptimizedBPETrainer::new(base.vocab);
trainer.initialize_from_corpus("./data/tamil_corpus.txt")?;
trainer.train(50000);
trainer.save_tokenizer_with_reserved_tokens("./output/tamil_tokenizer.json")?;

// Encode and decode
let (vocab, merges) = load_tokenizer("./output/tamil_tokenizer.json")?;
let encoder = FastBPEEncoder::new_with_merges(vocab, &merges)?;

let token_ids = encoder.encode_ordinary("வணக்கம்!");
let decoded = encoder.decode(&token_ids);
assert_eq!(decoded, "வணக்கம்!");
```

### Playground

![Partha Tamil Tokenizer Playground](assets/playground.png)

Launch the interactive tokenizer comparison playground:

```bash
uv run python playground/server.py
```

Then open `http://127.0.0.1:7860` in your browser.

## Results

| test | this repo | `indictok-256k` | mBERT |
|---|---:|---:|---:|
| bits/byte, wiki (lower is better) | **0.9193** | 1.0123 | 1.2575 |
| bits/byte, transcripts (lower is better) | **0.9150** | 0.9877 | 1.3266 |
| Tamil news accuracy, 6 classes | **0.5583** | 0.3932 | 0.3936 |

Bold marks the best score in each row.
Equal-token budget, three seeds.
Full tables, confidence intervals, per-class scores, and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

## The published models

The [Hub repository](https://huggingface.co/barryallen16/tamil-tokenizer-wiki) publishes two models.
Both are retrained as v4 with a dummy prefix on every training line.
The prefix closes the line-initial word split and is worth 0.5-1.3% depending on merge budget.

| file | merges | vocab | wiki B/tok | IndicCorp B/tok |
|---|---:|---:|---:|---:|
| `tamil_tokenizer_wiki.json` | 30,000 | 31,793 | 13.115 | 13.053 |
| `tamil_tokenizer_wiki_120k_mixed.json` | 120,000 | 122,723 | 14.720 | 16.251 |

The 120K mixed model is the default.
It ties the best specialist on each domain.
Going from 30K to 120K merges costs about 11% in throughput.
Use the 30K model when artifact size matters more than sequence length.

## Documentation

- [Results](docs/RESULTS.md): full tables, confidence intervals, byte-matched controls, and caveats.
- [Hugging Face model](https://huggingface.co/barryallen16/tamil-tokenizer-wiki): the published 30K and 120K tokenizers and the training corpora.
