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
| 4 | [Multi-Domain Training Corpus Scaling](#4-multi-domain-training-corpus-scaling) | measured; mix recommendation overturned |
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
| `export_hf` + `verify_hf_export.py` | roadmap asked for HF interop; it silently did not work | **99.4–99.7% ID agreement** with Python `tokenizers`, remainder categorised |
| IndicCorpV2 Tamil fetched, mixed corpus built | domain skew was assumed, never measured | by-domain table in Section 4 |
| Merge budget swept 30K/60K/120K | needed to know whether the mix was worth its dilution | mix **only wins above ~100K merges** |
| Upsampled-wiki control (×8) | separated "more data" from "more diverse data" | byte-identical model: volume is irrelevant |

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

**Corrected later:** at a 120K vocab the bottleneck does reappear, but it is the
`pair_to_merge` *hash lookup*, not the rescan — see Section 4, where stripping
the merge list shows the automaton costs 17% at 4x patterns while the merge
stage costs 2.4x. The right fix there is a flat or indexed pair table.

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

### What was built

```bash
python scripts/fetch_indiccorp.py --limit-mb 400      # 419 MB slice of IndicCorpV2 ta.txt
python scripts/prep_corpus.py data/indiccorp_ta.txt data/indiccorp
python scripts/prep_corpus.py --split data/tamil_wiki_clean.txt data/wiki
python scripts/prep_corpus.py --mix data/mixed 200 wiki+ic
```

The full `ai4bharat/IndicCorpV2` `data/ta.txt` is 11.8 GB, so `fetch_indiccorp.py`
pulls a leading byte range via HTTP `Range` requests (resumable, 3.8 MB/s).
The 419 MB slice is not degenerate: 97.6% unique lines, 77% hapax words,
31-36 words/line, stable across the whole file.

IndicCorp is one document per line, so `prep_corpus.py` splits documents into
sentences and applies the same cleaner and Tamil-dominance gate as the wiki
corpus. That keeps the two training sets comparable:

| Corpus | Train | Held out | Unique words found |
|---|---:|---:|---:|
| wiki | 25.2 MB (91,956 lines) | 5.5 MB (20,000) | 211,688 |
| IndicCorp | 199.4 MB (725,529 lines) | 13.8 MB (50,000) | 869,120 |
| mixed (11% wiki) | 224.6 MB | — | 960,132 |

Held-out sets are the *tail* of wiki and the *head* of IndicCorp — different
regions of each source, so neither model is scored on text adjacent to its own
training slice.

### Result: compression by domain

bytes/token, higher is better. `score.rs`, single-thread, held-out splits only.

| Training corpus | Merges | wiki held-out | IndicCorp held-out |
|---|---:|---:|---:|
| wiki 25 MB | 30K | **10.183** | 10.280 |
| IndicCorp 199 MB | 30K | 9.707 | **10.916** |
| mixed 25 wiki + 199 ic | 30K | 10.005 | 10.948 |
| wiki 25 MB | 60K | 10.622 | 10.773 |
| IndicCorp 199 MB | 60K | 10.151 | **11.416** |
| mixed 25 wiki + 199 ic | 60K | 10.509 | 11.549 |
| wiki 25 MB | 120K | 10.895 | 11.095 |
| IndicCorp 199 MB | 120K | 10.554 | 11.813 |
| mixed 25 wiki + 199 ic | 120K | **10.914** | **11.866** |

Three findings, one of which overturns this section's original recommendation.

**1. Domain skew is real and asymmetric.** At 120K merges the wiki model loses
6.1% on IndicCorp (11.095 vs 10.914 for mixed), and the IndicCorp model loses
3.3% on wiki. Wiki is the better general-purpose Tamil corpus: it is
homogeneous, so its frequent n-grams are core orthography that transfers.
IndicCorp is diverse news/web prose, so its top merges are more topic-bound.

**2. Volume is irrelevant; diversity is what costs.** A wiki-only model trained
on the same 25 MB repeated 8 times (202 MB) produces a **byte-identical** model:
10.183 / 10.280, to the last digit. The trainer keys on unique words, so
upsampling scales frequencies uniformly. This rules out "we just need more data"
as an explanation for anything in this table.

**3. The mix is only worth it at a large merge budget — the opposite of this
section's advice.** At 30K the mixed model is dominated: it loses to wiki-only
on wiki (10.005 vs 10.183) and ties IndicCorp-only on IndicCorp (10.948 vs
10.916). The extra vocabulary diversity dilutes a small merge budget. At 120K the
same mix **wins on both domains**. The original section recommended a mix with
no mention of budget, which would have been the wrong call for a 32K model.

### Cost: throughput at large vocab

| Model | Merges | Vocab | wiki MB/s | IndicCorp MB/s |
|---|---:|---:|---:|---:|
| wiki | 30K | 31,261 | 17.3 | 17.7 |
| IndicCorp | 30K | 31,690 | 17.0 | 18.9 |
| mixed | 30K | 31,934 | 16.9 | 18.7 |
| wiki | 120K | 106,042 | 5.5 | 7.3 |
| IndicCorp | 120K | 121,690 | 7.3 | 8.2 |
| mixed | 120K | 121,934 | 6.9 | 7.6 |

Going 30K → 120K buys 7-9% compression and costs **2.4x throughput**.

**Where that cost is.** Stripping the merge list and re-timing isolates it:

| Config | MB/s |
|---|---:|
| 31K vocab, merges stripped | 37.9 |
| 122K vocab, merges stripped | 31.5 |
| 31K vocab, merges on | 17.4 |
| 122K vocab, merges on | 7.3 |

The automaton is nearly free at 4x the patterns (-17%). The merge stage is what
costs 2.4x, and `profile_encode` confirms the pretokenizer is unchanged
(18.6 bytes/chunk in both). So the cost is the `pair_to_merge` hash table itself:
at 120K entries it no longer fits in cache and every pair lookup is a miss.

**This corrects the Phase 3 conclusion.** The rescan is indeed cheap, but at
large vocab the *lookup* is not. The upgrade path for Phase 3 is a flat or
indexed pair table, not a linked list with a heap.

### Recommended default

**30K merges on cleaned wiki** for interactive work: 17 MB/s and 10.18 bytes/token,
and it is within 1% of the best possible on IndicCorp anyway. **120K mixed** when
sequence length matters more than latency — it is the best model on both
domains, and its throughput problem is a known, isolated, fixable one.

### Not done

Project Madurai (literature/poetry), government/legal open data and technical
corpora are still unfetched. Publishing to Hugging Face needs repo credentials.

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

### Phase 2: Data Cleaning & Retraining — done except publish
- [x] MediaWiki cleaning in `scripts/extract_wiki_fast.py`
- [x] Clean 32K and 64K models trained on the cleaned corpus
- [x] Fetch 419 MB of IndicCorpV2 Tamil; build mixed corpus
- [x] Compression measured **by domain** on held-out splits
- [ ] Publish models to Hugging Face (needs repo credentials)

### Phase 3: Encoder Throughput — done, 1 rejected, 1 re-scoped
- [x] Packed `u64` pair keys
- [x] IDs carried through segmentation; no chunk slicing in the merge loop
- [x] Zero-copy `Vec<&str>` from `pretokenize`
- [x] One reused symbol buffer per text
- [x] rayon `encode_batch` / `encode_ordinary_batch`
- [x] `bench.rs`, `profile_encode.rs`, `score.rs`
- [ ] Doubly linked list + heap — **rejected**: 4.9 symbols/chunk makes the rescan cheap
- [ ] Criterion benches — **rejected**: covered without a new dependency
- [ ] Flat/indexed pair table — **re-scoped to large vocab**: 2.4x cost at 120K merges, see Section 4

### Phase 4: Ecosystem Compatibility — done
- [x] GPT-2 byte-to-unicode table, pinned by a test
- [x] `export_hf` → `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`
- [x] Export reproduces our pretokenizer (`Split`) and base clusters (291 promotion merges)
- [x] `verify_hf_export.py`: 99.4-99.7% ID agreement, remainder fully categorised

### Phase 5: Multi-domain training — done, next step is budget not corpus
- [x] IndicCorpV2 Tamil fetched and prepared
- [x] Compression by domain measured at 30K / 60K / 120K merges
- [x] Mix recommendation overturned: only helps above ~100K merges
- [ ] Fix the 120K merge-map cache cost (flat or indexed pair table)
- [ ] Add `<0x00>..<0xFF>` byte-fallback to the export so HF stops dropping unvocab characters
- [ ] Decide on Unigram with a measured A/B at fixed vocab size

---

## Known limits carried forward

1. **Latin-1 byte-stand-ins.** Under GPT-2's table, 162 byte tokens are also
   ordinary Latin-1 text (`²`, `é`, `ñ`). Our encoder routes those through the
   byte fallback and emits two tokens; HF matches the stand-in and emits one.
   Affects ~0.2% of Tamil lines. Closing it requires byte-mapping in a
   pretokenizer, which then breaks every raw-text merge.
2. **HF drops characters absent from the vocab.** With `unk_token: null` and no
   `byte_fallback`, HF BPE silently discards them; ours emits their UTF-8 bytes.
   Seen on lines containing Han characters (`艾青`) in an IndicCorp-trained model.
   Fix: add `<0x00>..<0xFF>` tokens, set `byte_fallback: true`, and prepend a
   `Replace` to the decoder. Affects ~0.04% of lines in the mixed corpus.
3. **Throughput is chunk-bound at 30K, merge-lookup-bound at 120K.** ~1.7M chunks
   per 32 MB means per-chunk fixed costs dominate at small vocab; past ~100K
   merges the `pair_to_merge` hash table stops fitting in cache and costs 2.4x.
   The two regimes need different fixes: the first wants fewer per-chunk costs,
   the second wants a flatter pair table.
4. **`model_max_length` must not go in `tokenizer.json`.** HF's schema rejects the
   field and the file becomes unloadable; it belongs in `tokenizer_config.json`.