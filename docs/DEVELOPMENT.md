# Development

How the code is organized, how encoding works, and the scripts that produce the measurements.

## Project structure

```
.
├── lib.rs                  # Library root - re-exports public types
├── vocab.rs                # TamilTokenizer - base vocabulary + byte mappings
├── model.rs                 # The canonical model both engines read
├── pretokenizer.rs         # TamilPretokenizer - regex-based text splitting
├── trainer.rs              # OptimizedBPETrainer - parallel BPE training
├── encoder.rs              # FastBPEEncoder - encode/decode over the canonical model
├── train.rs                # CLI: train a tokenizer from a corpus file
├── encode.rs               # CLI: encode text to token IDs
├── decode.rs               # CLI: decode token IDs back to text
├── benchmark.rs            # Corpus benchmark (Rust)
├── bench.rs                # Throughput: single-thread vs rayon, bytes/token
├── profile_encode.rs       # Where encode time goes: pretokenize vs merges
├── score.rs                # Bytes/token + throughput for one corpus (fast, in-process)
├── export_hf.rs            # CLI: write Hugging Face tokenizer artifacts
├── encode_lines.rs         # CLI: JSON id-per-line output (for verify scripts)
├── bench_encode.rs         # CLI: encode one text or stdin lines (for scripts)
├── bench_speed.rs          # CLI: JSON throughput report (for scripts)
├── test_special_tokens.rs  # Special token integration tests
├── main.rs                 # Interactive encoder/decoder REPL
├── tests/lib.rs            # Test suite (50 tests)
├── docs/
│   ├── RESULTS.md          # Full benchmark tables and caveats
│   ├── USAGE.md            # All CLI and Rust API examples
│   └── DEVELOPMENT.md      # Architecture, encoding, and scripts
├── scripts/evaluate.py     # Compression and fertility report (needs no extra deps)
├── scripts/extract_wiki_fast.py # Wiki dump extraction with MediaWiki cleaning
├── scripts/fetch_indiccorp.py    # Pull a byte range of IndicCorpV2 Tamil
├── scripts/prep_corpus.py        # Clean/split/mix corpora into train + held-out
├── scripts/verify_hf_export.py   # Checks HF export IDs against the Rust encoder
├── scripts/bench_field.py        # Scores this tokenizer against the field on one corpus
├── scripts/downstream_lm.py       # Trains one small model per tokenizer, reports bits per byte
├── scripts/colab_downstream.py    # The same, on a Colab GPU, with the corpora built there
├── scripts/classify_tamil.py      # Fine-tunes the saved models on six-class Tamil headlines
├── scripts/compare_classify.py    # Accuracy, macro F1 and the paired bootstrap for that task
├── scripts/fetch_transcripts.py   # Tamil video transcripts, cleaned and split by video
├── scripts/verify_install.py     # Checks the export loads through AutoTokenizer
├── scripts/vocab_utilization.py  # How much of a trained vocab is actually used
├── scripts/prefix_ab.py          # A/B the line-initial dummy-prefix fix
├── scripts/colab_bpb.py          # Bits-per-byte vs general-purpose tokenizers (GPU)
├── scripts/publish_hf.py         # Publish to the Hub with a measured model card
├── scripts/morph_alignment.py    # MorphScore: are our boundaries on morpheme edges?
├── scripts/spm_ab.py             # SentencePiece unigram-vs-bpe reference A/B
├── scripts/extract_wiki_text.py  # Legacy extractor, superseded by extract_wiki_fast.py
├── vocabcount.rs                 # Prints the base-vocab breakdown and checks it sums
└── data/tamil_corpus.txt   # 117-line sample corpus
```

### The measurement scripts

These exist because a published number stopped being true, or a published artifact stopped loading.
Each one re-derives its result rather than trusting a stored value:

| script | what it settles |
|---|---|
| `vocab_utilization.py` | How much of a trained vocab is used at all, and which merge ranks carry the weight |
| `prefix_ab.py` | Whether prepending a space to every training line helps. Runs both arms and prints ADOPT/REJECT |
| `colab_bpb.py` | Bits-per-byte against mBERT, Llama 3 and `cl100k_base`. Needs a GPU; runs in Colab |
| `publish_hf.py` | Re-measures before uploading, and refuses if decode fidelity is under 100% |
| `verify_install.py` | Whether the export loads through `AutoTokenizer` with bos, eos, pad and unk resolved |
| `bench_field.py` | How this tokenizer compares with the field on one corpus, same code path for everyone |
| `fetch_transcripts.py` | Brings in a third split: Tamil video captions, split by video |
| `downstream_lm.py` | Whether better compression gives a better model: one model, three tokenizers, bits per byte |
| `classify_tamil.py` | Whether the compression advantage survives a real task: six-class Tamil headline classification, same procedure per tokenizer |
| `compare_classify.py` | Accuracy, macro F1 and the paired bootstrap interval, for the token budget and the byte-matched control |

`cargo run --release --bin vocabcount` prints the base-vocabulary breakdown and asserts the buckets sum to the real vocab size.
That breakdown is maintained by hand and had drifted from the builder in `vocab.rs`.
It summed to 1,136 against an actual 1,135, which is where a stale "607 tokens" figure came from.

A caveat `colab_bpb.py` enforces on itself: a scoring LM's own tokenizer must never appear in the comparison table.
It was trained on its own segmentation, so it wins on that alone.
Qwen2.5 scores 0.77 bits/byte against our 1.15 in the same run, which says nothing about Tamil tokenization.

## Architecture

```
Text Input
    │
    ▼
┌──────────────────┐
│  Pretokenizer    │  Regex-based splitting
│                  │  (Tamil-aware Unicode patterns)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Canonical model │  One vocabulary and one
│                  │  merge list, shared with the
│                  │  Hugging Face export
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  BPE merges      │  Lowest-ranked pair first,
│                  │  every occurrence in one pass
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Token IDs       │  Output vector of u32 token IDs
└──────────────────┘
```

The BPE trainer uses a max-heap to select the most frequent token pair at each merge step.
It stores packed `u64` pair keys for cache-friendly access and loads the corpus in parallel with `rayon`.

## How encoding works

The trainer learns merges over raw text, so Tamil clusters and merged tokens are stored as raw Tamil strings.
`FastBPEEncoder` encodes each pretokenized chunk in three steps.
First it seeds the chunk with one symbol per character, because Hugging Face's `BPE` model has no seed symbols and starts from characters.
Then it applies the merges: the lowest-ranked adjacent pair first, and every occurrence of that pair in one pass.
Then it emits one token ID per symbol, where a character the vocab does not hold falls back to one token per UTF-8 byte, which keeps encoding lossless for any input.

The base clusters come back through the promotion merges.
There are 819 of them, they rank ahead of the learned merges, and they rebuild each cluster from its characters.
`model.rs` builds that list, and both the encoder and the exporter read it, so the two engines segment a chunk the same way.

Byte tokens are spelled `<0xNN>`, the way Hugging Face spells them, and a token that is not a byte token is decoded as literal text.
The vocab mixes raw-text tokens (Tamil clusters and merges, trained on raw text) with byte tokens, so a token like `café` is the word and not three bytes.

Special tokens keep the IDs stored in the file: a trained vocab already contains them, and the encoder reuses those IDs instead of remapping them.
The exact ID range where they start depends on the corpus, because the trainer mints IDs for clusters it discovers while reading.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `regex` | Unicode-aware pretokenization patterns |
| `rayon` | Parallel corpus processing |
| `serde` / `serde_json` | Vocab serialization |
| `rustc_hash` | Fast hash maps (FxHashMap) |
| `ahash` | Hash maps for trainer internals |
