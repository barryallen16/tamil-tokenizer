# Usage

The README has the three-command quickstart.
This file covers all six CLI steps, the Rust library API, and the Hugging Face export.

## Quickstart

### Pre-trained model

The Hugging Face repo **[barryallen16/tamil-tokenizer-wiki](https://huggingface.co/barryallen16/tamil-tokenizer-wiki)** publishes two trained tokenizers alongside the training corpora: 30K merges on cleaned wiki, and the recommended 120K merges on wiki + IndicCorpV2.

Load the 120K model with one line:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("barryallen16/tamil-tokenizer-wiki")
tok.encode("வணக்கம் இது தமிழ்").input_ids
```

The 120K model sits at the repository root.
The 30K model is in `tamil_tokenizer_wiki_hf/`:

```python
tok = AutoTokenizer.from_pretrained(
    "barryallen16/tamil-tokenizer-wiki", subfolder="tamil_tokenizer_wiki_hf"
)
```

Without `transformers`, use the `tokenizers` library and the `_hf/` file:

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tamil_tokenizer_wiki_120k_mixed_hf/tokenizer.json")
```

To work with the models from Rust, or to retrain, download this repo's own format:

```bash
REPO=https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main

# 30K on cleaned wiki (2.7 MB)
curl -L -o output/tamil_tokenizer_wiki.json $REPO/tamil_tokenizer_wiki.json

# 120K mixed (11.7 MB), the default
curl -L -o output/tamil_tokenizer_wiki_120k_mixed.json \
  $REPO/tamil_tokenizer_wiki_120k_mixed.json

# It loads with this repo's encoder:
cargo run --release --bin encode -- ./output/tamil_tokenizer_wiki.json "வணக்கம்"
# 11690 1980

# Download the 38 MB training corpus
curl -L -o data/tamil_wiki_corpus.txt $REPO/tamil_wiki_corpus.txt
```

The bare `.json` files use this repo's `{vocab, merges, metadata}` format and load through the Rust encoder.
The `_hf/` directories hold the Python-side conversion.
The [published models section](../README.md#the-published-models) covers what that conversion does and does not preserve.

This tokenizer adds no BOS or EOS. `tok.bos_token_id` and `tok.eos_token_id` are there for a training loop that wants them.

`scripts/evaluate.py` takes the tokenizer path as an argument and defaults to the local `output/tamil_tokenizer.json`.

### 1. Train a tokenizer

```bash
# Build the training binary
cargo build --release

# Train on a Tamil corpus with 50,000 merge operations
cargo run --release --bin train -- ./data/tamil_corpus.txt 50000 ./output/tamil_tokenizer.json
```

This writes a JSON vocab file to `output/tamil_tokenizer.json`.

### 2. Encode text

```bash
# Encode a string to token IDs
cargo run --release --bin encode -- ./output/tamil_tokenizer.json "வணக்கம்! இது தமிழ்"
```

### 3. Decode token IDs

```bash
# Decode token IDs back to text
cargo run --release --bin decode -- ./output/tamil_tokenizer.json 224 174 181
# வ
```

### 4. Interactive mode

```bash
# Launch interactive REPL (defaults to output/tamil_tokenizer.json)
cargo run --release --bin main

# Or specify a custom vocab path:
cargo run --release --bin main -- ./output/tamil_tokenizer.json
```

### 5. Run benchmarks

```bash
# Rust encoder benchmark
cargo run --release --bin benchmark

# Throughput on a real corpus: single-thread vs rayon, plus bytes/token
cargo run --release --bin bench -- ./output/tamil_32k.json ./data/tamil_wiki_clean.txt

# Where encode time goes
cargo run --release --bin profile_encode -- ./output/tamil_32k.json ./data/tamil_wiki_clean.txt

# Compression + throughput for one corpus, in-process
cargo run --release --bin score -- ./output/tamil_32k.json ./data/wiki/test.txt wiki

# Compression and fertility report (no extra dependencies)
python scripts/evaluate.py
python scripts/evaluate.py ./output/tamil_tokenizer.json
```

### 6. Export for Hugging Face

```bash
# Writes tokenizer.json, tokenizer_config.json, special_tokens_map.json,
# vocab.json, merges.txt
cargo run --release --bin export_hf -- ./output/tamil_64k.json ./output/hf64k

# Check the export loads through AutoTokenizer (needs: pip install transformers)
python scripts/verify_install.py ./output/hf64k

# Confirm Python produces the same IDs (needs: pip install tokenizers)
cargo build --release --bin encode_lines
python scripts/verify_hf_export.py ./output/tamil_64k.json ./output/hf64k
```

The third argument to `export_hf` sets `model_max_length`, the cap `transformers` applies when you pass `truncation=True`.
It is unlimited by default, so a published tokenizer does not cut a caller's input.

## Library usage (Rust)

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

### Encoding and decoding

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

### Special tokens

The encoder supports an o200k-style token layout:

| Range | Tokens |
|-------|--------|
| `0` to `base - 1` | Base vocabulary (Tamil clusters, bytes, etc.) |
| `base` to `base + 7` | Named specials: `<\|startoftext\|>`, `<\|endoftext\|>`, `<\|pad\|>`, `<\|unk\|>`, `<\|mask\|>`, `<\|question\|>`, `<\|answer\|>`, `<\|code\|>` |
| `base + 8` to `base + 107` | 100 reserved tokens: `<\|reserved_0\|>` to `<\|reserved_99\|>` |

Here `base` is `base_vocab_size`.

```rust
// Check if a token ID is special
let is_special = encoder.is_special_token(token_id);

// Get ID of a named special token
let bos_id = encoder.get_special_token_id("<|startoftext|>");
```

## Hugging Face export

`export_hf` writes the artifacts `transformers` expects:

| File | Purpose |
|------|---------|
| `tokenizer.json` | `PreTrainedTokenizerFast`: `Split` pretokenizer + BPE model |
| `tokenizer_config.json` | `model_max_length`, BOS/EOS/PAD/UNK names |
| `special_tokens_map.json` | The same four special tokens, for `transformers` |
| `vocab.json` | Slow-tokenizer vocab (`token -> id`) |
| `merges.txt` | Slow-tokenizer merges, rank order, `#version: 0.2` header |

`model_max_length` is unlimited unless you pass a third argument to `export_hf`. `transformers` reads it as the cap it applies when you pass `truncation=True`, and a base tokenizer should not cut a caller's input.

Two design choices have no HF equivalent, and the exporter translates both instead of leaving them to fail silently in Python:

- The pretokenizer. This repo uses a Tamil-aware regex.
  HF's `ByteLevel` pretokenizer hardcodes GPT-2's pattern, and that pattern byte-maps text before BPE sees it, so no merge in the file could ever match.
  The export uses `Split` with our own regex, which leaves raw text intact.
- Base clusters. Our trainer treats `கே` and `ங்` as atomic base symbols.
  HF BPE has no base-symbol set: it starts from single characters and can only reach a token through merges.
  The exporter adds 819 promotion merges that build each such token from its characters, ranked by decreasing token length so HF resolves overlaps by longest match the way the base vocabulary does.
  It also appends the 4 characters (`ோ`, `ூ`, `ொ`, `ௌ`) that the base vocab lacks but the corpus contains.

Measured against `data/wiki/test.txt`, `verify_hf_export.py` reports 100% decode fidelity and 100% ID agreement.
The two engines read one model and apply the same merges, so the exporter needs no runtime bridge.

### Known limits

ID agreement with the Rust encoder is 100% on both held-out splits. The two engines read the same vocabulary and the same merge list, and they apply the same merges: the lowest-ranked adjacent pair first, and every occurrence of that pair in one pass.

That took a change to how the encoder segments.
It used to seed a chunk with a greedy longest match over the base clusters, which Hugging Face's `BPE` model cannot express, since that model has no seed symbols and starts from characters.
The exporter bridges the difference with its 819 promotion merges, and the encoder now starts from characters too.
The cost is about 1.7% more tokens on Tamil, because the promotion rules and a greedy longest match do not always draw the cluster boundaries in the same place.
The published artifact already behaved this way; the Rust encoder now agrees with it.

Two fixed defects are worth recording because both were worse than their old descriptions:

- Latin-1 corruption. Running `ByteLevel` after `ByteFallback` re-applied the GPT-2 table to already-decoded output, so a decoded `é` became raw byte 0xE9 and came back as U+FFFD.
  Across 3,988 Latin-1-bearing lines in four corpora, 2,074 (52%) decoded corrupted before the fix and none did after.
  The decoder now uses `ByteFallback` alone.
- Dropped characters. `export_hf` emits `<0x00>..<0xFF>` with `byte_fallback: true`, so unvocab characters are no longer dropped and Han text round-trips correctly.

`verify_hf_export.py` reports decode fidelity and ID agreement separately because they catch different failures.
The publish script gates on decode fidelity (100%), not ID agreement.
