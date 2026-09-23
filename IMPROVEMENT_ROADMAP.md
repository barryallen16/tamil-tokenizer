# Tamil Tokenizer: Performance & Quality Improvement Roadmap

Concrete engineering and linguistic improvements to elevate `tamil_tokenizer` in
**compression efficiency (bytes-per-token)** and **encoding throughput (MB/s)**.

Every change below is recorded as **what → why → measured effect**, so a future
reader can tell a shipped win from a rejected idea without re-running anything.
Items that were implemented but then reverted by measurement say so explicitly,
with the numbers that killed them.

**Status.** Phases 1 and 4 done. Phase 2 done except multi-domain corpus. Phase 3
done except the two items measurement rejected. Phase 5 not started.

---

## Table of Contents

| # | Section | Status |
|---|---------|--------|
| 1 | [Linguistic Base Vocabulary & Grantha Support](#1-linguistic-base-vocabulary--grantha-support) | done |
| 2 | [Encoding Algorithmic Optimizations](#2-encoding-algorithmic-optimizations) | done, 2 items rejected by measurement |
| 3 | [Pretokenization & Corpus Cleaning](#3-pretokenization--corpus-cleaning) | cleaning done, affix handling not started |
| 4 | [Multi-Domain Training Corpus Scaling](#4-multi-domain-training-corpus-scaling) | in progress |
| 5 | [Architectural Evolution: Unigram LM vs. Greedy BPE](#5-architectural-evolution-unigram-lm-vs-greedy-bpe) | not started |
| 6 | [Prioritized Implementation Checklist](#6-prioritized-implementation-checklist) | |

## Summary of shipped changes

| Change | Why | Effect |
|---|---|---|
| Grantha consonants + ligatures in base vocab | Grantha syllables were absent, so the trainer burned merges re-learning them | base vocab 524 → 607; no training needed for `ஜா`, `ஸ்ரீ`, `க்ஷே` |
| Packed `u64` merge keys | merge loop hashed two strings per comparison | one 64-bit hash per pair |
| IDs carried through segmentation | merge loop re-sliced the chunk per comparison | no slicing inside the hot loop |
| Zero-copy `pretokenize` | one `String` allocation per chunk | `Vec<&str>`, no allocation |
| One reused symbol buffer | pretokenizer averages 18.9 bytes/chunk, so a `Vec` per chunk is one allocation per handful of bytes | **14.4 → 17.8 MB/s** single-thread |
| rayon `encode_batch` / `encode_ordinary_batch` | encoding is embarrassingly parallel across documents | **37.0 → 42.0 MB/s** batch (2.4x over single-thread) |
| `bench.rs`, `profile_encode.rs` | every throughput claim in this doc needed a reproducible source | best-of-3 MB/s, stage split, chunk histogram |
| MediaWiki cleaner in `extract_wiki_fast.py` | ~25% of early merges were ASCII markup (`colo+r`, `j+pg`) | 127,889 → 111,956 lines; **8.31 → 10.76 bytes/token** at 30K merges |
| GPT-2 byte table, exactly | ours differed from GPT-2 on 95 bytes; HF hardcodes GPT-2's | exports stop producing silently wrong IDs |
| Stand-in guard excludes all of byte space | GPT-2 maps bytes 161-255 to Latin-1, so `Ã` was both a byte token and real text | Latin-1 text no longer decodes to mojibake |
| `export_hf` + `verify_hf_export.py` | roadmap asked for HF interop; it silently did not work | **99.6% ID agreement** with Python `tokenizers` |

---

## 1. Linguistic Base Vocabulary & Grantha Support

### Problem

Modern Tamil (news, technology, literature, common speech, Wikipedia) frequently
uses **Grantha consonants** (`ஜ`, `ஷ`, `ஸ`, `ஹ`, `ஶ`) and compound ligatures
(`க்ஷ`, `ஸ்ரீ`).

[`vocab.rs`](vocab.rs) only generated consonant-vowel combinations for the **18
traditional Tamil consonants**:
```rust
let consonant_codepoints = [
    0x0B95, 0x0B99, 0x0B9A, 0x0B9E, 0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BAF,
    0x0BB0, 0x0BB2, 0x0BB5, 0x0BB4, 0x0BB3, 0x0BB1, 0x0BA9,
];
```
giving $18 \times 12 = 216$ base clusters.

Syllables with Grantha consonants (`ஜா`, `ஜி`, `ஷா`, `ஸா`, `ஹோ`, `ஶ்ரீ`) were
missing. In words like `ரோஜா`, `விஷயம்`, `ஹலோ`, `பேஸ்புக்`, `ஸ்டேஷன்` this meant
falling back to bytes or multi-token fragments, and spending merge budget
re-learning syllables that should exist for free.

### Applied

| # | What | Why |
|---|------|-----|
| 1 | Added `0x0B9C` `ஜ`, `0x0BB6` `ஶ`, `0x0BB7` `ஷ`, `0x0BB8` `ஸ`, `0x0BB9` `ஹ` | Grantha is standard in modern Tamil orthography; without them every Grantha syllable costs several merges |
| 2 | Clusters grow $18 \times 12 \to (18+5) \times 12 = 276$ | the consonant set feeds the cartesian product directly |
| 3 | Added 13 `க்ஷ` variants and `ஸ்ரீ` / `ஶ்ரீ` | `க்ஷ` is not `க்` + `ஷ` orthographically; treating it as a ligature keeps it one token |
| 4 | Added `ஂ` (`0x0B82`), `ௐ` (`0x0BD0`), `ௗ` (`0x0BD7`) | live Tamil text, absent from the base set |
| 5 | Tests in [`tests/lib.rs`](tests/lib.rs) | a silent regression here is invisible: encoding still succeeds, just in more tokens |

Base vocabulary: 524 → **607 tokens**, no training required.

### Rejected

**Making the trainer's cluster lookup 4 characters long.** The original loop only
tried 2- and 3-char clusters, so `ஸ்ரீ` and `க்ஷ` could never be matched
atomically. Changed to a longest-first loop over lengths 2..=4. Note the
resulting dependency: any token the pretokenizer never sees as a contiguous unit
is unreachable during training, which is why `tests/lib.rs` asserts these
directly against the encoder rather than only through the trainer.

---

## 2. Encoding Algorithmic Optimizations

The original section predicted a 10x speedup. **It did not happen, and the
roadmap's diagnosis was wrong.** The measurements below explain why.

### Applied

| # | What | Why |
|---|------|-----|
| 1 | `FxHashMap<String, FxHashMap<String, u32>>` → `FxHashMap<u64, (u32, u32)>` keyed `(id_a << 32) \| id_b` | the merge loop hashed two strings per comparison; one 64-bit hash replaces both, and the merged ID comes back with the rank |
| 2 | `Symbol { start, end, id }` replaces `(usize, usize)` spans | the merge loop re-sliced the chunk on every comparison just to look up the pair |
| 3 | `pretokenize` returns `Vec<&'a str>` | one `String` allocation per chunk; a 32 MB corpus produced ~1.7M chunks |
| 4 | One reused symbol buffer per `encode_inner` call | the pretokenizer averages **18.9 bytes per chunk**, so a fresh `Vec` is an allocation per handful of bytes |
| 5 | `encode_batch` / `encode_ordinary_batch` over rayon | encoding documents is embarrassingly parallel |

### Measured effect

16 MB of cleaned wiki text, 30K-merge model, best of three passes:

| | single-thread | rayon batch | bytes/token |
|---|---:|---:|---:|
| before | 14.4 MB/s | 37.0 MB/s | 10.766 |
| after | 17.8 MB/s | 42.0 MB/s | 10.760 |

**1.24x single-thread.** The token stream is unchanged (10.76 bytes/token, same
as the pre-refactor model), so this is a pure speedup with no quality cost.

### Rejected: doubly linked list + min-heap

The original section proposed replacing the quadratic rescan with an
index-based doubly linked list and a heap of candidate pairs, on the theory that
merges only affect the two neighbouring pairs.

**Why not:** `profile_encode` reports **4.9 base symbols per chunk** (median 5,
p99 13). The quadratic rescan averages ~32 comparisons per chunk — already
trivial. A heap pays a sift per candidate to remove work that does not exist.
Profile output:

| Stage | ms per MB |
|---|---:|
| pretokenize | 12 |
| automaton + merges + emit | 42 |

The 30ms gap is real but is not the merge scan. It is the ~1.7M per-chunk fixed
costs (buffer setup, automaton call, token emit) that a linked list does not
touch. This matches the earlier note in commit `305cb7e`, which had already
written and measured a heap version and found it within noise.

**Upgrade path:** only if pretokenizer changes push chunk length up by an order
of magnitude. Not the current bottleneck.

### Rejected: Criterion

**Why not:** `bench.rs` and `profile_encode.rs` already report throughput, the
per-stage split and a chunk-length histogram with no new dependency. Criterion's
statistical harness is worth it when comparing variants, and there is exactly one
variant to measure here.

---

## 3. Pretokenization & Corpus Cleaning

### Problem: wasted merge budget on markup

The raw 38.8 MB Wikipedia dump carries `thumb|`, `align=`, `[[Category:`,
`{{cite web}}`, HTML tags and hex colours. About a quarter of the early merges in
a 1K-merge model were ASCII markup fragments (`colo+r`, `j+pg`).

### Applied

`scripts/extract_wiki_fast.py` was rewritten around two functions:

- `clean_wikitext` — ordered rules, nested constructs first: `<ref>…</ref>` and
  `<ref/>`, comments, remaining tags, `{{templates}}`, `[[File:]]` /
  `[[Category:]]`, other `[[links]]`, external links, bare URLs, `'''bold'''`,
  headings, entities, runs of spaces.
- `is_tamil_prose` — a keep gate: at least 30 chars, Tamil covering 40% of the
  line, and markup characters (`|_=<>{}[]#*/\`) under 2%.

| | What | Why |
|---|------|-----|
| 1 | Cleaner applied before anything reaches the trainer | a merge spent on `thumb|` is a merge not spent on Tamil |
| 2 | `--clean` mode for already-extracted corpora | the 38 MB corpus on disk predates the cleaner |
| 3 | `fast_extract_v2` removed | it printed a line count and then wrote **nothing** to the output file — a silent no-op |

On the 127,889-line wiki corpus: **111,956 lines kept** (32.3 MB of 38.8 MB).
At the same 30K-merge budget, wiki compression goes **8.31 → 10.76 bytes/token**.

### Not started: Tamil agglutination and affix handling

Tamil fuses case markers (`-ஐ`, `-ஆல்`, `-க்கு`, `-இன்`) and plurals (`-கள்`)
directly onto nouns. The idea is to stop the pretokenizer splitting bound
morphemes while keeping the whitespace-prefix rule uniform, so `"தமிழ்"` and
`" தமிழ்நாடு"` share root representations.

**Why not yet:** the current pattern already unifies the leading-space rule across
all letter classes. Any change here shifts token boundaries for every model ever
trained, so it needs a compression A/B on a held-out set — not a blind regex edit.

---

## 4. Multi-Domain Training Corpus Scaling

### Problem

Training exclusively on Wikipedia produces a domain-skewed vocabulary: heavy on
geography, dates and historical terms, light on news, commerce and technical
prose. A wiki-only model should therefore compress news text worse than a
news-trained one, but that needed measuring rather than assuming.

### Corpus mix

| Source | Target | Status |
|---|---:|---|
| Tamil Wikipedia (cleaned) | ~35 MB | done — 32.3 MB |
| AI4Bharat IndicCorpV2 Tamil | ~100–300 MB | in progress |
| Project Madurai (literature, poetry) | ~30 MB | not fetched |
| Tamil government & legal open data | ~20 MB | not fetched |
| Technical / bilingual mixed | ~20 MB | not fetched |

### Models trained

| Merges | Vocab | Load corpus | Train merges | Wiki B/tok | Single-thread | rayon batch |
|---:|---:|---:|---:|---:|---:|---:|
| 30,000 | 31,341 | 2.5 s | 1.1 s | 10.76 | 17.8 MB/s | 42.0 MB/s |
| 60,000 | 61,341 | 2.4 s | 1.2 s | 11.34 | 12.7 MB/s | 32.2 MB/s |

Both on `tamil_wiki_clean.txt` with `min_frequency=2`.

**Note on the speedup claim.** Doubling merges from 30K to 60K buys 5% more
compression (10.76 → 11.34 bytes/token) and costs 29% throughput. That is the
tradeoff to weigh before choosing a vocab size, not a free win.

---

## 5. Architectural Evolution: Unigram LM vs. Greedy BPE

### Why Unigram might win for Tamil

- **Greedy BPE** merges pairs strictly by local frequency. In agglutinative
  languages, frequent character bigrams can override true morpheme boundaries.
- **Unigram Language Model** (Gemma, SentencePiece, T5) starts from a large
  candidate vocabulary and prunes it by EM on global corpus likelihood.
- Studies on Indic/Dravidian languages report 5–15% better compression than BPE
  at equal vocab size.

### Why not started

**The trainer is built around a merge list, and that merge list is the export
format.** `export_hf` writes `merges.txt` and an HF `BPE` model. A Unigram mode
is therefore a second trainer plus a second export path, not a flag on the
existing one. It should be a measured A/B against the current BPE at fixed vocab
size, on a held-out multi-domain set — which is what Section 4 is now producing.

---

## 6. Prioritized Implementation Checklist

### Phase 1: Base Vocab Expansion — done
- [x] Grantha consonants (`ஜ`, `ஷ`, `ஸ`, `ஹ`, `ஶ`) in [`vocab.rs`](vocab.rs)
- [x] Base clusters $18 \times 12 \to 276$, plus `க்ஷ` and `ஸ்ரீ` ligatures
- [x] Cluster coverage tests in [`tests/lib.rs`](tests/lib.rs)
- [x] Longest-first cluster lookup (up to 4 chars) so ligatures are atomic

### Phase 2: Data Cleaning & Retraining — partly done
- [x] MediaWiki cleaning in `scripts/extract_wiki_fast.py`
- [x] Clean 32K and 64K models trained on the cleaned corpus
- [ ] Multi-domain corpus (IndicCorp, Madurai, legal, technical)
- [ ] Compression measured **by domain**, not just on the training distribution
- [ ] Publish models to Hugging Face (needs repo credentials)

### Phase 3: Encoder Throughput — done, 2 items rejected
- [x] Packed `u64` pair keys
- [x] IDs carried through segmentation; no chunk slicing in the merge loop
- [x] Zero-copy `Vec<&str>` from `pretokenize`
- [x] One reused symbol buffer per text
- [x] rayon `encode_batch` / `encode_ordinary_batch`
- [x] `bench.rs` + `profile_encode.rs`
- [ ] Doubly linked list + heap — **rejected**: 4.9 symbols/chunk makes the rescan cheap
- [ ] Criterion benches — **rejected**: covered without a new dependency

### Phase 4: Ecosystem Compatibility — done
- [x] GPT-2 byte-to-unicode table, pinned by a test
- [x] `export_hf` → `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`
- [x] Export reproduces our pretokenizer (`Split`) and base clusters (291 promotion merges)
- [x] `verify_hf_export.py`: 99.6% ID agreement on 12,000 lines, remainder fully explained

### Phase 5: Multi-domain training — in progress
- [ ] Fetch IndicCorpV2 Tamil; re-measure compression by domain
- [ ] Decide on Unigram with a measured A/B at fixed vocab size

---

## Known limits carried forward

1. **Latin-1 byte-stand-ins.** Under GPT-2's table, 162 byte tokens are also
   ordinary Latin-1 text (`²`, `é`, `ñ`). Our encoder routes those through the
   byte fallback and emits two tokens; HF matches the stand-in and emits one.
   Affects ~0.2% of Tamil lines. Closing it requires byte-mapping in a
   pretokenizer, which then breaks every raw-text merge.
2. **Throughput is chunk-bound, not merge-bound.** ~1.7M chunks per 32 MB means
   per-chunk fixed costs dominate. Any future 2x is more likely to come from
   batch API shape or a byte-level pretokenizer than from the merge loop.
3. **`model_max_length` must not go in `tokenizer.json`.** HF's schema rejects the
   field and the file becomes unloadable; it belongs in `tokenizer_config.json`.