# Tamil Tokenizer

A BPE tokenizer library for Tamil text, written in Rust.
It trains on custom Tamil corpora, encodes quickly, and exports to Hugging Face without changing token IDs.
On the same 20M-token pretraining budget it is 10.1-10.2% better than `indictok-256k` and 36.7-36.9% better than mBERT in bits per byte on Tamil wiki text, and it is 16.5 points more accurate on six-class Tamil news classification.

## Features

- Tamil-aware base vocabulary of 1,135 tokens with no training: 256 byte tokens, 12 independent vowels, 23 standalone consonants, 276 consonant+vowel-sign clusters (including Grantha `ஜ ஶ ஷ ஸ ஹ`), 4 special characters (`ஃ ஂ ௐ ௗ`), 15 compound ligatures (`க்ஷ`, `ஸ்ரீ`), 528 conjunct stems, 10 Tamil digits, and 11 extra characters.
- The conjunct stems are what moved compression. Tamil joins consonants constantly (`க்ஷ`, `ஸ்ரீ`, `க்ட`, `ண்ட`), and each one used to cost 3-4 tokens.
- Training runs in parallel through [rayon](https://docs.rs/rayon) and uses packed pair storage plus a priority queue for merge selection.
- Encoding uses a packed merge table with a per-pair rule cache, so a pair is looked up once per chunk instead of once per merge.
- `encode_batch` and `encode_ordinary_batch` split a document set across all cores.
- `export_hf` writes `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, and `merges.txt`. The output is checked against the Python `tokenizers` and `transformers`.
- The Rust encoder and the Hugging Face export read the same vocabulary and merge list, so they emit the same token IDs for the same text.
- Special tokens cover BOS/EOS, language tags, question and answer markers, and 100 reserved slots in an o200k-style layout.
- Byte mapping uses the exact GPT-2 byte-to-unicode table, so the export matches Hugging Face's `ByteLevel`.

## Quickstart

### Playground

Launch the interactive tokenizer comparison playground modeled after `tiktokenizer`:

```bash
uv run --with "transformers,tokenizers,tiktoken" python playground/server.py
```

Then open `http://127.0.0.1:7860` in your browser.
Compare Tamil Tokenizer (120K and 30K) against OpenAI (o200k, cl100k), Google Gemma 4, Alibaba Qwen 3.8, Meta Llama 3.3, IndicBERT, and Sarvam in real time.

### Python

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("barryallen16/tamil-tokenizer-wiki")
tok.encode("வணக்கம் இது தமிழ்").input_ids
```

This needs `pip install transformers`.

The 120K model is at the repository root; the 30K model is in `tamil_tokenizer_wiki_hf/`.
See [The published models](#the-published-models) for the difference.

### Rust

```bash
# Train on the 117-line sample corpus
cargo run --release --bin train -- ./data/tamil_corpus.txt 50000 ./output/tamil_tokenizer.json

# Encode with the model you just trained
cargo run --release --bin encode -- ./output/tamil_tokenizer.json "வணக்கம்!"
```

All six CLI steps, the Rust library API, and the Hugging Face export are in [docs/USAGE.md](https://github.com/barryallen16/tamil-tokenizer/blob/main/docs/USAGE.md).

## Results

| test | this repo | `indictok-256k` | mBERT |
|---|---:|---:|---:|
| bits/byte, wiki (lower is better) | **0.9193** | 1.0123 | 1.2575 |
| bits/byte, transcripts (lower is better) | **0.9150** | 0.9877 | 1.3266 |
| Tamil news accuracy, 6 classes | **0.5583** | 0.3932 | 0.3936 |

Bold marks the best score in each row. Equal-token budget, three seeds.

On wiki bits/byte this repo is 10.1-10.2% better than `indictok-256k` and 36.7-36.9% better than mBERT; on transcripts the margins are 7.9-8.1% and 44.9-45.0%.
On six-class Tamil news headlines this repo scores 0.5583 against 0.3932 and 0.3936, a gap of 16.5 points; the byte-matched control scores 0.5601 against 0.4574 and 0.3888.
Full tables, confidence intervals, per-class scores, and caveats are in [docs/RESULTS.md](https://github.com/barryallen16/tamil-tokenizer/blob/main/docs/RESULTS.md).

## The published models

The [Hub repo](https://huggingface.co/barryallen16/tamil-tokenizer-wiki) publishes two models.
Both are retrained as v4 with a dummy prefix on every training line.
The prefix closes the line-initial word split and is worth 0.5-1.3% depending on merge budget.

| file | merges | vocab | wiki B/tok | IndicCorp B/tok |
|---|---:|---:|---:|---:|
| `tamil_tokenizer_wiki.json` | 30,000 | 31,793 | 13.115 | 13.053 |
| `tamil_tokenizer_wiki_120k_mixed.json` | 120,000 | 122,723 | 14.720 | 16.251 |

The 120K mixed model is the default. It ties the best specialist on each domain.
After the merge-table re-indexing, going from 30K to 120K costs about 11% in throughput rather than the 2.4x an earlier revision quoted.
Use the 30K model when artifact size matters more than sequence length.

Each model is published three ways: the bare `.json` in this repo's `{vocab, merges, metadata}` format, a `_hf/` directory holding the Python-side conversion, and, for the 120K model, a copy at the repository root.
`AutoTokenizer.from_pretrained` loads the root copy, so the one-liner needs no subfolder argument.
The bare `.json` files only load through the Rust encoder; `Tokenizer.from_file` raises a JSON parse error on them.

`scripts/publish_hf.py` generates the model card at publish time.
It refuses to upload unless decode fidelity is 100% on both held-out splits and the export loads through `AutoTokenizer` with its special tokens resolved.

## Documentation

- [Results](https://github.com/barryallen16/tamil-tokenizer/blob/main/docs/RESULTS.md): full tables, confidence intervals, byte-matched controls, and caveats.
- [Usage](https://github.com/barryallen16/tamil-tokenizer/blob/main/docs/USAGE.md): all CLI commands, the Rust API, and the Hugging Face export.
- [Development](https://github.com/barryallen16/tamil-tokenizer/blob/main/docs/DEVELOPMENT.md): project layout, architecture, encoding internals, and the measurement scripts.
- [Hugging Face model](https://huggingface.co/barryallen16/tamil-tokenizer-wiki): the published 30K and 120K tokenizers and the training corpora.
