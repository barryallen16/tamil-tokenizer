# Tamil Tokenizer: Performance & Quality Improvement Roadmap

This document outlines concrete engineering and linguistic improvements to elevate `tamil_tokenizer` to state-of-the-art performance in both **compression efficiency (fertility/bytes-per-token)** and **runtime encoding throughput (MB/s)**.

Status: Phases 1, 2 (partly) and 4 are done. Phase 3 is done except for the two
items that measurement rejected, recorded below with their numbers.

---

## Table of Contents
1. [Linguistic Base Vocabulary & Grantha Support](#1-linguistic-base-vocabulary--grantha-support) — done
2. [Encoding Algorithmic Optimizations](#2-encoding-algorithmic-optimizations) — partly done, see measurements
3. [Pretokenization & Corpus Cleaning](#3-pretokenization--corpus-cleaning) — cleaning done, affixes not
4. [Multi-Domain Training Corpus Scaling](#4-multi-domain-training-corpus-scaling) — not started
5. [Architectural Evolution: Unigram LM vs. Greedy BPE](#5-architectural-evolution-unigram-lm-vs-greedy-bpe) — not started
6. [Prioritized Implementation Checklist](#6-prioritized-implementation-checklist)

---

## 1. Linguistic Base Vocabulary & Grantha Support

### Problem

Modern Tamil (news, technology, literature, common speech, Wikipedia) frequently uses **Grantha consonants** (`ஜ`, `ஷ`, `ஸ`, `ஹ`, `ஶ`) and compound ligatures (`க்ஷ`, `ஸ்ரீ`).

Before, [`vocab.rs`](vocab.rs) only generated consonant-vowel combinations for the **18 traditional Tamil consonants**:
```rust
let consonant_codepoints = [
    0x0B95, 0x0B99, 0x0B9A, 0x0B9E, 0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BAF,
    0x0BB0, 0x0BB2, 0x0BB5, 0x0BB4, 0x0BB3, 0x0BB1, 0x0BA9,
];
```
This produced $18 \times 12 = 216$ base clusters.

Syllables formed with Grantha consonants (e.g. `ஜா`, `ஜி`, `ஷா`, `ஷி`, `ஸா`, `ஹா`, `ஹி`, `ஶ்ரீ`) were **missing from the base vocabulary**. When these appeared in words like `ரோஜா`, `விஷயம்`, `ஹலோ`, `பேஸ்புக்`, or `ஸ்டேஷன்`, the tokenizer:
1. Fell back to individual bytes or multi-token fragments, degrading compression.
2. Wasted valuable BPE merge operations during training just to reconstruct basic syllables that should exist natively in the base vocabulary.

### Applied

1. **Added Grantha consonants** to `vocab.rs`: `0x0B9C` (`ஜ`), `0x0BB6` (`ஶ`), `0x0BB7` (`ஷ`), `0x0BB8` (`ஸ`), `0x0BB9` (`ஹ`). The cartesian product grew from $18 \times 12 = 216$ to $(18 + 5) \times 12 = 276$ clusters.
2. **Added common ligatures**: 13 `க்ஷ` variants and `ஸ்ரீ` / `ஶ்ரீ`.
3. **Added** `ஂ` (`0x0B82`), `ௐ` (`0x0BD0`) and `ௗ` (`0x0BD7`).
4. **Tests**: `tests/lib.rs` asserts single-token encoding for each Grantha cluster and ligature, plus round-trip for `ரோஜா`, `ஹலோ`, `விஷயம்`, `ஸ்டேஷன்`, `பேஸ்புக்`, `ஸ்ரீராம்`, `க்ஷேமம்`.

Base vocabulary: 524 → **607 tokens**, no training required.

---

## 2. Encoding Algorithmic Optimizations

### Bottleneck A: nested maps and quadratic rescan in `apply_merges`

`apply_merges` compared pairs by slicing the chunk and hashing two strings per comparison.

#### Applied

1. **Packed integer lookup.** `FxHashMap<String, FxHashMap<String, u32>>` → `FxHashMap<u64, (u32, u32)>` keyed by `(id_a << 32) | id_b`, carrying `(rank, merged_id)`. One hash of a `u64` per pair instead of two string hashes, and the merged ID comes back with it.
2. **IDs resolved during segmentation.** `segment` now carries `Symbol { start, end, id }` instead of byte spans, so the merge loop never re-slices the chunk.
3. **Zero-copy pretokenization.** `TamilPretokenizer::pretokenize` returns `Vec<&'a str>` instead of `Vec<String>`, removing one heap allocation per chunk.
4. **Scratch buffer reuse.** The pretokenizer averages 18.9 bytes per chunk, so a `Vec` per chunk was one allocation per handful of bytes. `encode_inner` reuses one buffer for the whole text.

Measured on 16 MB of cleaned wiki text with the 30K-merge model:

| | single-thread | rayon batch |
|---|---:|---:|
| before | 14.4 MB/s | 37.0 MB/s |
| after | 17.8 MB/s | 42.0 MB/s |

#### Not applied: doubly linked list + min-heap

The roadmap proposed replacing the rescan with an index-based doubly linked list
and a heap of candidate pairs. **Measured and rejected.** `profile_encode`
reports 4.9 base symbols per chunk on average (median 5, p99 13), so the
quadratic rescan averages ~32 comparisons per chunk. A heap adds a sift per
candidate and buys nothing at that size.

Profiling (`cargo run --release --bin profile_encode`) also shows where the time
actually goes, and it is not the merge scan:

| Stage | ms per MB |
|---|---:|
| pretokenize | 12 |
| automaton + merges + emit | 42 |

Shortest honest summary: this is a 1.24x single-thread improvement, not the
10x this section predicted. The corpus that motivated the estimate produces
~1.7M chunks per 32 MB, so per-chunk fixed costs dominate and the rescan is
not the bottleneck.

### Bottleneck C: multi-threaded batch encoding

#### Applied

```rust
pub fn encode_batch<T: AsRef<str> + Sync>(&self, texts: &[T]) -> Vec<Vec<u32>>
pub fn encode_ordinary_batch<T: AsRef<str> + Sync>(&self, texts: &[T]) -> Vec<Vec<u32>>
```

Both use rayon. Measured 2.4-2.5x over single-thread on this machine.

### Benchmarks

`cargo run --release --bin bench -- <tokenizer> <corpus> [bytes]` reports
single-thread vs rayon throughput and bytes/token. `profile_encode` reports the
pretokenize / encode split.

### Criterion

Not adopted. `bench.rs` and `profile_encode.rs` cover the same ground with no
new dependency, and Criterion's statistical harness adds nothing to a
single-number throughput comparison.

---

## 3. Pretokenization & Corpus Cleaning

### Problem: wasted merge budget on markup

The raw 38.8 MB Wikipedia dump contains substantial markup (`thumb|`, `align=`, `[[Category:`, `{{...}}`, HTML tags, hex colors). About a quarter of the early merges in a 1K-merge model were ASCII markup fragments (`colo+r`, `j+pg`).

### Applied

`scripts/extract_wiki_fast.py` was rewritten around `clean_wikitext` (ordered
rules: `<ref>`/`<ref/>`, comments, remaining tags, `{{templates}}`, `[[File:]]`
and `[[Category:]]`, other `[[links]]`, external links, bare URLs, `'''bold'''`,
headings, entities) plus `is_tamil_prose`, a gate requiring Tamil to dominate and
markup characters to stay under 2%. `fast_extract_v2`, which wrote nothing to
the output file, was removed.

```bash
python scripts/extract_wiki_fast.py --clean data/tamil_wiki_corpus.txt data/tamil_wiki_clean.txt
```

On the 127,889-line wiki corpus this keeps 111,956 lines (32.3 MB of 38.8 MB).
At the same 30K-merge budget, compression goes from **8.31 to 10.76
bytes/token** on wiki text.

### Not applied: Tamil agglutination and affix handling

Refining the pretokenizer to keep bound morphemes together while unifying
whitespace prefixes is untouched. The current pattern already unifies the
leading-space rule across the letter classes, and any change here shifts token
boundaries for every model. It belongs with a compression A/B measurement on a
held-out set, not as a blind regex edit.

---

## 4. Multi-Domain Training Corpus Scaling

### Recommended Training Dataset Mix:

1. **Tamil Wikipedia** (Cleaned): ~35 MB — done (32.3 MB, `tamil_wiki_clean.txt`)
2. **AI4Bharat IndicCorp Tamil** (News, articles, web): ~100–300 MB — not fetched
3. **Project Madurai** (Classical & modern literature, poetry): ~30 MB — not fetched
4. **Tamil Government & Legal Open Data**: ~20 MB — not fetched
5. **Technical / Mixed Corpora** (Bilingual Tamil + English technical text): ~20 MB — not fetched

Only Wikipedia is present locally. Items 2-5 need downloads that were not part
of this pass, and the domain-skew problem is still open: the 32K model is
trained entirely on encyclopedic prose.

### Target Vocab Sizes

| Model | Merges | Vocab | Wiki B/tok | Status |
|---|---:|---:|---:|---|
| 32K | 30,000 | 31,341 | 10.76 | trained, exported |
| 64K | 60,000 | 61,341 | 11.34 | trained, exported |

Both trained on `data/tamil_wiki_clean.txt` with `min_frequency=2` in about 3.6 s
each. Publishing to Hugging Face needs repo credentials and was not done here.

---

## 5. Architectural Evolution: Unigram LM vs. Greedy BPE

### Why Unigram for Tamil?

- **Greedy BPE** merges pairs strictly based on local frequency. In agglutinative languages, frequent character bigrams often override true linguistic morpheme boundaries.
- **Unigram Language Model** (used by Google Gemma, SentencePiece, T5) initializes a large candidate vocabulary and iteratively prunes tokens using Expectation-Maximization (EM) based on global corpus likelihood.
- Empirical studies on Indic and Dravidian languages show that **Unigram models achieve 5–15% better compression and fertility** than BPE at the same vocabulary size.

Not started. The trainer is built around a merge list, and BPE merge lists are
the export format `export_hf` writes, so a Unigram mode is a second trainer plus
a second export path rather than a flag.

---

## 6. Prioritized Implementation Checklist

### Phase 1: Base Vocab Expansion (Linguistic Win) — done
- [x] Add Grantha consonants (`ஜ`, `ஷ`, `ஸ`, `ஹ`, `ஶ`) to [`vocab.rs`](vocab.rs).
- [x] Add base clusters $(18 + 5) \times 12 = 276$ and common ligatures (`க்ஷ`, `ஸ்ரீ`).
- [x] Update unit tests in [`tests/lib.rs`](tests/lib.rs) to verify new cluster coverage.

### Phase 2: Data Cleaning & Retraining (Quality Win) — partly done
- [x] Add robust MediaWiki cleaning to `scripts/extract_wiki_fast.py`.
- [x] Train a clean 32K and 64K model on the cleaned corpus.
- [ ] Curate a 100+ MB multi-domain Tamil corpus (IndicCorp + Madurai + legal + technical).
- [ ] Publish the trained models to Hugging Face.

### Phase 3: Encoder Throughput Refactoring (Speed Win) — partly done
- [x] Convert `apply_merges` to packed `u64` pair keys.
- [x] Carry resolved IDs through segmentation so the merge loop never re-slices.
- [x] Change `TamilPretokenizer::pretokenize` to return zero-copy `Vec<&str>` slices.
- [x] Reuse one symbol buffer per text instead of allocating per chunk.
- [x] Add `encode_batch` with Rayon multi-threading.
- [x] Add `bench.rs` and `profile_encode.rs` to measure throughput and stage costs.
- [ ] Doubly linked list neighbor updates — measured, rejected. Chunk size makes the rescan cheap; see Section 2.
- [ ] `benches/` using Criterion — skipped, existing binaries cover it without a new dependency.

### Phase 4: Ecosystem Compatibility — done
- [x] Match GPT-2's byte-to-unicode table exactly, with a test that pins it.
- [x] Add an export tool converting our format to Hugging Face `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt`.
- [x] Reproduce our pretokenizer and base clusters in the export (`Split` pretokenizer + 291 promotion merges).
- [x] Verify full interoperability with Python `tokenizers` — 99.6% ID agreement on 12,000 lines, with every remaining line attributable to the documented Latin-1 byte-stand-in limit.

### Phase 5: Multi-domain training (next)
- [ ] Fetch IndicCorp Tamil and Project Madurai; re-measure compression by domain.
- [ ] Decide on Unigram training with a measured A/B against the current BPE.