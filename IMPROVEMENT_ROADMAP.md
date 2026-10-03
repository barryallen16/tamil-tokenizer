# Tamil Tokenizer: Performance & Quality Improvement Roadmap

Concrete engineering and linguistic improvements to elevate `tamil_tokenizer` in
**compression efficiency (bytes-per-token)** and **encoding throughput (MB/s)**.

Every change below is recorded as **what → why → measured effect**, so a future
reader can tell a shipped win from a rejected idea without re-running anything.
Items that were implemented but then reverted by measurement say so explicitly,
with the numbers that killed them.

**Status.** Phases 1-5 done apart from the two open publishing items in Section
6. Phase 5's Unigram question is answered: measured, it is not worth building.
Two defects in the BPE trainer surfaced while answering it, and together they
were worth **25% of the token count** — the very quantity this document is about
— so every bytes/token figure below has been re-measured. Pre-fix numbers are
kept alongside the corrected ones wherever they were the basis of a conclusion.

A later audit found a second class of error: figures that were never
re-measured because nobody re-ran them. The export's **99.4–99.7% ID agreement
did not reproduce** (79.2% / 73.0%, and identical on `v2` and `v3`), and the
Latin-1 defect described as a ~0.2% token-count nuisance was in fact **52%
silent data corruption** on affected lines. Both are corrected in place, with
the original claims struck rather than deleted, and the cause recorded as
[Known Limit 6](#known-limits-carried-forward). Where a claim could not be
reproduced, it says so.

---

## Table of Contents

| # | Section | Status |
|---|---------|--------|
| 1 | [Linguistic Base Vocabulary & Grantha Support](#1-linguistic-base-vocabulary--grantha-support) | done |
| 2 | [Encoding Algorithmic Optimizations](#2-encoding-algorithmic-optimizations) | done, 2 items rejected by measurement |
| 3 | [Pretokenization & Corpus Cleaning](#3-pretokenization--corpus-cleaning) | cleaning done, affix handling not started |
| 4 | [Multi-Domain Training Corpus Scaling](#4-multi-domain-training-corpus-scaling) | measured; all figures corrected |
| 5 | [Architectural Evolution: Unigram LM vs. Greedy BPE](#5-architectural-evolution-unigram-lm-vs-greedy-bpe) | done; Unigram rejected on measurement |
| 6 | [Prioritized Implementation Checklist](#6-prioritized-implementation-checklist) | |
| 7 | [Open Work](#7-open-work) | proposed, not measured |

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
| `export_hf` + `verify_hf_export.py` | roadmap asked for HF interop; it silently did not work | **100% decode fidelity; 79.2% ID agreement** with Python `tokenizers` — see Known Limit 6 |
| IndicCorpV2 Tamil fetched, mixed corpus built | domain skew was assumed, never measured | by-domain table in Section 4 |
| Merge budget swept 30K/60K/120K | needed to know whether the mix was worth its dilution | mix **only wins above ~100K merges** |
| Upsampled-wiki control (×8) | separated "more data" from "more diverse data" | byte-identical model: volume is irrelevant (re-verified after the fix) |
| Trainer heap: re-push a stale entry instead of dropping it | a dropped entry is a pair the trainer can never see again, and the heap pops the *frequent* pairs first, so it discards exactly the best candidates | **10.156 → 13.468 bytes/token** at 30K, i.e. 25% fewer tokens |
| Trainer: sort unique words before minting token ids | ids were minted while walking a hash map, so the model depended on ahash's per-process seed | two runs on one corpus now produce byte-identical files |
| `spm_ab.py`: dropped `line.strip()` on the SentencePiece side | `score.rs` counts lines as written, so the two columns had different denominators | no-op on the cleaned corpora, but no longer flatters one side |
| `spm_ab.py`: read bytes/token instead of rebuilding it from the MB column | `score.rs` prints MB to one decimal, so `int(float(mb) * 1024 * 1024)` overshot by up to 0.05 MB | our row was inflated 0.4% (13.525 → 13.468), i.e. in our own favour |

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

Base vocabulary: 524 → **607 tokens**, no training required. A later commit
(conjunct stems, `46ac590`) took the same base set to **1,135** — 607 plus 529
consonant-virāma-consonant stems, less one duplicate — which is the figure
`tests/lib.rs` pins.

### Rejected

**Making the trainer's cluster lookup 4 characters long.** The original loop only
tried 2- and 3-char clusters, so `ஸ்ரீ` and `க்ஷ` could never be matched
atomically. Changed to a longest-first loop over lengths 2..=4. Note the
resulting dependency: any token the pretokenizer never sees as a contiguous unit
is unreachable during training, which is why `tests/lib.rs` asserts these
directly against the encoder rather than only through the trainer.

### Decided: line-initial words train in two forms and only one is reachable

The pretokenizer attaches one leading space to a chunk (` ?` in the pattern), so
mid-line ` கணிதம்` is a single chunk. The **first chunk of a line has no space**,
so it enters the trainer as `கணிதம்` — a different symbol sequence with its own
pair statistics. Only the spaced form survives training:

| token | in the trained vocab |
|---|---|
| ` கணிதத்தில்`, ` கட்டிடக்கலையில்`, ` இவர்`, ` ஒரே` | yes |
| `கணிதத்தில்`, `கட்டிடக்கலையில்`, `இவர்`, `ஒரே` | no |

So `கணினி` costs 1 token mid-line (` கணி` + `னி`, or 1 for words the merge list
covers) and 3 at the start of a line. SentencePiece gets this normalization for
free from `add_dummy_prefix`; we do not have an equivalent.

**Cost, measured with the fixed trainer:** retraining on a corpus with one space
prepended to every line removes **1.0% of tokens** (422,093 vs 426,412 over 5,000
held-out lines; 13.605 vs 13.468 bytes/token on the same byte denominator). Small
— 87% of lines start with a letter, but the average line-initial word costs only
~1 token — and it was 3.7% before the trainer fix, because the buggy trainer was
wasting merges re-attaching spaces to its own base clusters.

**Decided: ADOPT the dummy prefix for all newly trained models.** The A/B asked
for here has been run — `scripts/prefix_ab.py`, two 30K models on the same
corpus, scored on the held-out split:

| training corpus | bytes/token |
|---|---:|
| as-is | 13.468 |
| one space prepended to every line | **13.294** |

**+1.29%**, above the 0.5% threshold, and independently reproducible against the
estimate above. No pretokenizer change is needed: the trainer reads a corpus
file, so prepending the space to every *training* line gives SentencePiece's
`add_dummy_prefix` effect. Scoring stays on the **unprefixed** held-out split,
because that is what real input looks like.

This applies to **new models only**. Chunk boundaries shift, so the two
published artifacts and every model in `output/` remain on the old scheme. The
gain is small enough that republishing for 1.29% is not worth invalidating
existing ids; retrain when there is a reason to.

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
**This first table is the corrected one** — `v3_*` models, fixed trainer.

| Training corpus | Merges | wiki held-out | IndicCorp held-out |
|---|---:|---:|---:|
| wiki 25 MB | 30K | **13.468** | 13.449 |
| IndicCorp 199 MB | 30K | 12.781 | **14.747** |
| mixed 25 wiki + 199 ic | 30K | 13.138 | 14.729 |
| wiki 25 MB | 60K | 14.429 | 14.517 |
| IndicCorp 199 MB | 60K | 13.856 | 15.956 |
| mixed 25 wiki + 199 ic | 60K | 14.236 | 15.939 |
| wiki 25 MB | 120K | 15.149 | 15.330 |
| IndicCorp 199 MB | 120K | 14.829 | 16.952 |
| mixed 25 wiki + 199 ic | 120K | 15.212 | **16.944** |

**Every cell moved, and the reason is a trainer defect, not this section.** The
original table is kept below, because it is what this section's conclusions were
drawn from. It was measured with a merge heap that dropped stale entries (see
Section 5), so every published cell is 31-43% lower than the corrected one.

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

**How far these tables can be compared cell by cell.** Re-scoring the surviving
`v2_*` files with the current binary reproduces the published *wiki* rows to
within 0.3% (wiki 30K: 10.156 vs 10.183), but the IndicCorp and mixed rows drift
by 1.9% and 3.0%. Those rows were therefore taken on an earlier model
generation, and the shipped mix of corpus preparation and base vocabulary has
moved since.

The clean before/after is the pair measured here with one binary, which is what
the fix row in the summary quotes: **10.156 → 13.468** bytes/token at wiki 30K,
i.e. 25% fewer tokens. Across the whole grid the ranking of the nine
configurations is **unchanged** and the scale moves by 31-43%; representative
cells: wiki 30K on wiki held-out 10.183 → 13.468 (+32.3%), mixed 120K on
IndicCorp held-out 11.866 → 16.944 (+42.8%).

Keeping the ranking intact is worth stating plainly: it means the three findings
below were conclusions about the *corpus*, not about the bug. But each of their
figures had to be recomputed, and one of them changed shape.

Three findings, one of which overturns this section's original recommendation.

**1. Domain skew is real and asymmetric.** At 120K merges the wiki model loses
9.5% on IndicCorp (15.330 vs 16.944 for mixed), and the IndicCorp model loses
2.5% on wiki (14.829 vs 15.212). Wiki is the better general-purpose Tamil corpus:
it is homogeneous, so its frequent n-grams are core orthography that transfers.
IndicCorp is diverse news/web prose, so its top merges are more topic-bound.
The asymmetry got *sharper* with the fix (9.5% vs 6.1%), so this finding is
stronger than it was, not weaker.

**2. Volume is irrelevant; diversity is what costs.** A wiki-only model trained
on the same 25 MB repeated 8 times (202 MB) produces a **byte-identical** model:
13.468 / 13.449, to the last digit — re-verified against the fixed trainer, same
identical merge-list hash. The trainer keys on unique words, so upsampling scales
frequencies uniformly. This rules out "we just need more data" as an explanation
for anything in this table.

**3. The mix is only worth it at a large merge budget — the opposite of this
section's advice, but weaker than it looked.** At 30K the mixed model loses to
wiki-only on wiki (13.138 vs 13.468, -2.5%) and ties IndicCorp-only on IndicCorp
(14.729 vs 14.747). The extra vocabulary diversity dilutes a small merge budget.
At 120K the same mix **matches** the best specialist on each domain — 15.212 vs
15.149 on wiki and 16.944 vs 16.952 on IndicCorp, i.e. within 0.4%.

The corrected measurement is a **tie at 120K, not a win**. The previous table
showed mixed ahead on both domains (10.914 vs 10.895, 11.866 vs 11.813), but
those margins were 0.2-0.5% and the corrected ones are 0.4% and -0.05%, so the
honest reading is parity either way. What survives unchanged is the direction:
the mix is the right choice at 120K and the wrong one at 30K. The original
section recommended a mix with no mention of budget, which would have been the
wrong call for a 32K model.

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

**The absolute numbers in this table no longer reproduce, and it is not the
trainer fix.** Commit `60a4855` ("index merge rules by first token in csr layout
instead of hashing the pair", 10:31) implemented the recommendation this
subsection ends with, and it landed *after* these rows were taken. Re-measured
today with the current binary, the same pre-fix 120K model runs at **27.1 MB/s,
not 5.5** — a 5x correction that no compression change can explain.

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

**This corrected the Phase 3 conclusion, and Phase 3 then fixed it.** The rescan
is indeed cheap, but at large vocab the *lookup* was not, and the upgrade path
was a flat or indexed pair table rather than a linked list with a heap. That
shipped as `60a4855`; the section was never updated to say so.

Re-measured on the corrected models, same binary, best of three:

| Model | Merges | Vocab | wiki MB/s | IndicCorp MB/s |
|---|---:|---:|---:|---:|
| wiki | 30K | 31,793 | 26.5 | 28.0 |
| IndicCorp | 30K | 32,218 | 26.3 | 26.4 |
| mixed | 30K | 32,462 | 26.6 | 25.5 |
| wiki | 120K | 121,793 | 23.6 | 24.9 |
| IndicCorp | 120K | 122,218 | 22.9 | 23.9 |
| mixed | 120K | 122,462 | 23.4 | 24.1 |

**30K → 120K now costs ~12% throughput, not 2.4x**, while buying 13-26%
compression. The large-vocab penalty this subsection was written about is gone,
and the trade it warned against is now clearly worth taking.

### Recommended default

**120K merges on the mixed corpus.** It matches the best specialist on both
domains (15.212 wiki / 16.944 IndicCorp), which is 13% ahead of 30K wiki on wiki
held-out and 26% ahead on IndicCorp, and after the CSR change it costs only ~12%
throughput (23.4 vs 26.5 MB/s).

This **reverses** the previous recommendation, which said 30K wiki for interactive
work and treated 120K as a specialist option carrying a known throughput problem.
Both halves of that reasoning are now wrong: the throughput problem is fixed, and
30K wiki is no longer within 1% of the best model on IndicCorp (13.449 vs 16.944,
which is 26%).

**30K merges on cleaned wiki** is still the pick when latency or artifact size
matters more than sequence length: 26.5 MB/s and 13.47 bytes/token.

### Not done

Project Madurai (literature/poetry), government/legal open data and technical
corpora are still unfetched. Publishing to Hugging Face needs repo credentials.

---

## 5. Architectural Evolution: Unigram LM vs. Greedy BPE

### The argument this section was built on

- **Greedy BPE** merges pairs strictly by local frequency. In agglutinative
  languages, frequent character bigrams can override true morpheme boundaries.
- **Unigram Language Model** (Gemma, SentencePiece, T5) starts from a large
  candidate vocabulary and prunes it by EM on global corpus likelihood.
- Studies on Indic/Dravidian languages report 5–15% better compression than BPE
  at equal vocab size. **This figure was never sourced, and it does not
  reproduce.** The measured value is 0.6% — see below.

### Result: 0.6%, so we are not building it

`scripts/spm_ab.py` trains SentencePiece `unigram` and `bpe` on the same corpus
at the same vocabulary, then scores both against ours. SentencePiece is the
reference implementation here rather than a hand-rolled one; see "Rejected"
below for the cost of getting that wrong.

32K vocab, trained on `data/wiki/train.txt`, `byte_fallback=True`,
`character_coverage=1.0`, identity normalization, same seed size, held-out
splits:

| model | wiki held-out | IndicCorp held-out |
|---|---:|---:|
| sentencepiece unigram | 13.933 (1.70 tok/word) | 13.803 (1.79) |
| sentencepiece bpe | 13.844 (1.71) | 13.711 (1.81) |
| ours (bpe, `v3_wiki_30000`) | 13.468 (1.76) | 13.422 (1.84) |

The wiki column is the whole held-out file. The IndicCorp column is the first
6 MB of it, which is `spm_ab.py`'s default cap — all three models see the same
text so the comparison holds, but it is a slightly different subset from
Section 4's full-file 13.449 for our model. An earlier run of this table showed
ours at 13.525 because the harness rebuilt bytes from `score.rs`'s MB column,
which is printed to one decimal; see the summary table.

**Unigram beats BPE by 0.64% on wiki and 0.67% on IndicCorp.** Two consequences,
and the second is the useful one:

1. **Do not build a Unigram trainer.** 0.6% does not pay for a second trainer and
a second export path, which is what this section always said it would cost.
2. **The gap worth closing is in our own BPE, not in the objective.** Ours is
**2.8% behind SentencePiece's BPE** (13.844 vs 13.468) using the *same* greedy
algorithm on the *same* corpus — larger than the 0.6% Unigram would buy, and
fixable in the existing trainer rather than by a rewrite. That is exactly where
the two defects below were found.

Scope, stated plainly: this measures compression only. The usual arguments for
Unigram are morphological alignment and downstream model quality, and neither is
measurable without training a model, so neither was tested. That omission is not
theoretical — [TamilTok](https://aclanthology.org/2026.dravidianlangtech-1.7/)
(DravidianLangTech 2026) measured it on Tamil and found UnigramLM **consistently
better than BPE and WordPiece on morphological alignment at every vocabulary size,
with no fertility penalty**. So the correct reading of this section is narrower
than "Unigram lost": on the one axis we measured, it wins by 0.6% and is not worth
a second trainer; on the axis we did not measure, the Tamil-specific evidence
favours it, and a morphology-aware vocabulary (Section 7.4) is one way to chase
that without writing a Unigram trainer at all.

### Rejected: a hand-rolled Unigram trainer

A `unigram.rs` was written and deleted before it was ever committed. It reached
**5.579 bytes/token** on wiki held-out against a BPE measuring 10.156 — worse
than the baseline it was meant to beat, by 1.8x. The reasons were recorded in
the session log and are worth keeping:

- It used **hard Viterbi counts** in the E-step. SentencePiece uses
  **forward-backward marginals** (`Lattice::PopulateMarginal`).
- It **pruned by raw frequency**. SentencePiece scores each piece by the
  likelihood actually lost if that piece is removed, i.e. an approximate loss
  `F * (logprob_sp - logprob_alt)` per piece, with pieces that have no second-best
  segmentation kept unconditionally (`unigram_model_trainer.cc`).

Frequency pruning keeps the short pieces and drops the long ones, so the model
collapsed toward characters: 26.6% of emitted tokens were single characters, and
Viterbi preferred `'க'` at log p -1.8 over a whole word at -12. The 1.8x deficit
was the implementation, not the objective.

**The lesson to carry forward:** do not implement a reference algorithm without
an oracle. The same corpus scored through SentencePiece would have shown the gap
in minutes, and it is what eventually settled the question.

### Found while running this A/B: two trainer defects

The A/B needed our BPE as a baseline, and the baseline was 36% behind
SentencePiece's BPE. That gap was not the objective and not the corpus:

1. **The merge heap dropped stale entries.** When a pair's frequency fell, its
   heap entry still held the old count. The pop loop discarded such entries
   rather than re-pushing them with the corrected count, and since only pairs
   touching the newly merged token were ever re-pushed, a discarded pair was gone
   for good. The heap pops the *highest* counts first, so the pairs it discarded
   were exactly the frequent ones it should have merged. **10.156 → 13.468
   bytes/token** (25% fewer tokens) at 30K, and 44,642 corrections over 30,000
   merges. See `trainer.rs` and the brute-force oracle in `tests/lib.rs`.
2. **Training was not reproducible.** Token ids are minted while walking the
   unique-word map, and the merge tie-break compares packed ids, so iterating an
   `AHashMap` made the model depend on ahash's per-process seed. Two runs on one
   corpus produced different merge lists. Fixed by sorting before assigning ids.

Both shipped with regression tests, and both tests were confirmed to fail when
the defect is reintroduced.

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
- [x] Flat/indexed pair table — shipped as `60a4855` (first-token CSR layout); the 2.4x cost at 120K merges it targeted is gone, see Section 4

### Phase 4: Ecosystem Compatibility — done
- [x] GPT-2 byte-to-unicode table, pinned by a test
- [x] `export_hf` → `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`
- [x] Export reproduces our pretokenizer (`Split`) and base clusters (819 promotion merges)
- [x] `verify_hf_export.py`: **100% decode fidelity**, 79.2% ID agreement, remainder fully categorised
- [x] Re-verified against `v3_*` — the 99.4-99.7% figure on the `v2` generation did not reproduce (Known Limit 6)
- [x] Latin-1 corruption fixed: `ByteLevel` removed from the decoder sequence (`201f135`)

### Phase 5: Multi-domain training — done
- [x] IndicCorpV2 Tamil fetched and prepared
- [x] Compression by domain measured at 30K / 60K / 120K merges, corrected for the trainer fix
- [x] Mix recommendation refined: matches the best specialist at 120K, loses 2.5% at 30K
- [x] 120K merge-map cache cost fixed — first-token CSR layout, commit `60a4855`
- [x] Unigram decided by measured A/B: **no** — 0.6% does not pay for a second trainer
- [x] Trainer defects the A/B exposed fixed: 25% of the token count, two regression tests
- [x] Add `<0x00>..<0xFF>` byte-fallback to the export so HF stops dropping unvocab characters (`20117fb`)

---

## Known limits carried forward

1. ~~**Latin-1 byte-stand-ins.**~~ **Closed.** This entry was wrong about the
   mechanism and understated the damage. It claimed HF "matches the stand-in and
   emits one" token; in fact `ByteLevel` was running *after* `ByteFallback` and
   re-applying the GPT-2 table to already-decoded output, so a decoded `é`
   (U+00E9, itself a stand-in for byte 0xE9) was turned back into the raw byte
   0xE9 and decoded to U+FFFD. That is silent corruption, not a token count.
   Swept across 3,988 Latin-1-bearing lines in four corpora: **2,074 (52.0%)
   corrupted before, 0 after** (`201f135`). No pretokenizer change was needed, so
   no raw-text merge was broken — the entry's stated blocker never applied.
2. ~~**HF drops characters absent from the vocab.**~~ **Already fixed, entry was
   stale.** `export_hf` has emitted `<0x00>..<0xFF>` with `byte_fallback: true`
   since `20117fb`, and Han text round-trips correctly. What remains is only that
   our byte-token *ids* differ from HF's: we use the base-vocab stand-in chars,
   HF uses `<0xNN>`. Same token count, different ids — folded into Known Limit 6.
3. **Throughput is chunk-bound at 30K; the 120K merge-lookup penalty is fixed.**
   ~1.7M chunks per 32 MB means per-chunk fixed costs dominate at small vocab.
   The 120K regime now costs ~12% rather than 2.4x, after `60a4855` replaced the
   hashed pair lookup with a first-token-indexed table.
4. **Line-initial words are a separate, unreachable form.** The pretokenizer
   attaches one leading space per chunk, so the first word of every line trains as
   `கணிதம்` while every other occurrence is ` கணிதம்`, and only the spaced form
   earns merges. Measured cost of the split: ~1.0% of tokens. `scripts/prefix_ab.py`
   runs the A/B; closing it shifts every chunk boundary and so invalidates every
   model, which is why it is recorded rather than fixed — see Section 3.
5. **`model_max_length` must not go in `tokenizer.json`.** HF's schema rejects the
   field and the file becomes unloadable; it belongs in `tokenizer_config.json`.
6. **HF export is a lossy approximation, not an equivalent.** This entry
   supersedes the 99.4-99.7% agreement figure this document used to publish,
   which did not reproduce: **79.2%** on wiki held-out and **73.0%** on IndicCorp
   held-out, and *identical on `v2` and `v3`*, so it was never true for the
   generation it was attributed to. Decode fidelity is 100%.

   The cause is structural, not a data defect: all 120,000 learned merges survive
   the export (verified — zero lost). HF's `BPE` model always starts from
   individual characters, while `FastBPEEncoder` starts from the 1,135 base
   clusters and only then applies merges by rank. The same merge list over
   different starting symbols yields different segmentations: on ` வள்ளச்சி`
   ours yields one token, HF yields ` வள்ள` + `ச்சி`. Registering base clusters
   as `added_tokens` was tried and is worse — HF extracts added tokens before
   the BPE model runs, merges never fire, and agreement drops to 0%.

   In token *count* the gap is small: 1.565 vs 1.539 tok/word on cleaned wiki
   held-out, ~1.7%. The 79.2% is ID agreement, not token count. HF `BPE` has no
   seed-symbol concept, so this cannot be closed inside the format — treat the
   export as an approximation and quote our own encoder's numbers.

---

## 7. Open Work

Sections 1-6 record what shipped and what measurement killed. This section is a
different kind of thing: **everything below is proposed, not measured.** Each
item carries the source it came from and the measurement that would settle it
instead of an effect size. Sources are external unless marked as ours.

### 7.1 Finish what is already open

| Item | Why | What settles it |
|---|---|---|
| ~~Re-run `verify_hf_export.py` against `v3_*`~~ | **done** — the 99.4-99.7% figure did not reproduce; 79.2% wiki / 73.0% IndicCorp, identical on `v2`, cause recorded in Known Limit 6 | — |
| ~~`<0x00>..<0xFF>` byte-fallback in the export~~ | **done, earlier than this section claimed** (`20117fb`); Han round-trips, no [tokenizers#1996](https://github.com/huggingface/tokenizers/issues/1996) effect | — |
| Re-verify the Latin-1 decoder fix on IndicCorp held-out for all three `v3_*` models | measured on 3,988 Latin-1 lines corpus-wide and on `v3_mixed` end-to-end; the other two were only checked on wiki held-out | 100% decode on both splits for all three |
| Publish an artifact for the new default | Section 4 recommends 120K mixed, but `output/` is gitignored so that recommendation exists only as prose | a named artifact plus the exact reproducible training command |
| Hugging Face publication | blocked on repo credentials, unchanged. Note the published 32K export predates `201f135` and still carries the Latin-1 corruption | — |

### 7.2 Evaluation: we have exactly one metric

bytes/token is the only thing this document optimises, and the literature is
explicit that it is not a reliable objective by itself:

- **Minimising corpus token count does not reliably improve downstream
  performance** (Schmidt et al. 2024).
- **Rényi efficiency can be raised arbitrarily while degrading model quality**
  (Cognetta et al. 2024), so "we win an intrinsic metric" is not evidence on its own.
- **Fertility and parity are not reliably predictive** of downstream results
  (Ali et al. 2024).

Those three are quoted from TokEval's related work rather than read as primary
sources; treat them as pointers, not as settled findings.

[TokEval](https://arxiv.org/html/2608.18062v1) (Meister, EPFL, 2026) proposes a
suite that goes past fertility and compression: UTF-8 character-boundary
integrity, digit place-value alignment, line-break handling, and multilingual
parity. Its controlled pretraining sweep found information-theoretic metrics
predict language-model ability (Spearman |ρ| up to 0.80), while
structure-sensitive metrics correlate with task accuracy.

1. Adopt TokEval (`cimeister/tokenizer-intrinsic-evals`) and report its metrics
   next to bytes/token. Report **fertility (tokens/word)** as the headline
   number, because that is what Indic tokenizer papers publish.
2. Add **MorphScore** ([Arnett & Bergen 2025](https://aclanthology.org/2025.coling-main.441/),
   [`catherinearnett/morphscore`](https://github.com/catherinearnett/morphscore),
   datasets for 22 languages in the paper, more in the repo since): it measures
   how often token boundaries land on morpheme boundaries. Read the source before
   treating it as a target, though, because that paper's own conclusion is the
   opposite of what this item originally assumed. It finds **some** evidence that
   tokenization quality explains the agglutinative-vs-fusional performance gap,
   and **none for the role of morphological alignment**, attributing the gap
   instead to training-data size once scaled by the "byte-premium". So use
   MorphScore as a descriptive instrument — *how morphologically plausible are our
   boundaries?* — and not as a proxy for downstream quality. Tamil boundary labels
   come from TamilTok's **TamilMorph**, below.
3. Stand up a **downstream smoke test**. [CHiPSAL 2026](https://aclanthology.org/2026.chipsal-1.19/)
   uses Tamil news-title classification with a small BERT; anything comparable
   would let a tokenizer change be checked for effect instead of assumed. For any
   LM, report **bits-per-byte**, which is tokenizer-agnostic.
4. **Benchmark against a general multilingual tokenizer.** **Measured** — see
   the table below; this was the item the project existed to move and it had
   never been run on our own splits. `scripts/colab_bpb.py` scores our export,
   mBERT and Llama 3 against the same held-out text under one fixed LM
   (Qwen2.5-0.5B), so the only variable is the tokenizer.

   Scored on the **cleaned** wiki held-out split (20,000 lines, 5.76 MB),
   32K-merge model:

   | | tok/word | bits/byte |
   |---|---:|---:|
   | ours | **1.77** | **1.06** |
   | mBERT | 3.74 | 2.16 |

   **2.11x better on fertility, 51% lower bits-per-byte.** This pair is
   trustworthy because neither tokenizer is the scoring model's own, so neither
   row gets the home advantage a native tokenizer gets. mBERT's 3.74
   tok/word independently corroborates the 3.64 TamilTok reports for it.

   Corpus choice moved this result and would have flattered us if picked the
   other way. On the raw MediaWiki dump ours scores 1.946 and mBERT 3.241,
   because the dump is ~9% ASCII markup and mBERT handles ASCII cheaply. Clean
   prose is the honest test, and it is where mBERT is *worse* (3.735).

   Two things this does **not** say. It is the 32K-merge model, not the 120K
   default Section 4 recommends, so it is a floor. And Llama 3 is gated while
   `cl100k_base` and Sarvam are still unrun. A model's *own* tokenizer must
   never appear in this table: Qwen2.5 scored its own segmentation at 0.77
   bits/byte against our 1.15 in the first run, which is a property of the
   training pair, not of Tamil tokenization.

**Sanity check on the one number we do have.** Our fertility is 1.76 tok/word at
30K and 1.56 at 120K. Published Tamil fertility for Indic-specialised tokenizers
is 1.86 (VerChol, 32,991 vocab) and 1.9969 (Gnani). That puts us in the right
range, but the test sets are entirely different, so this is a sanity check and
not a win.

### 7.3 Corpus

**Deduplication.** Our pipeline has a cleaner and a Tamil-dominance gate but no
dedup. The BharatGen tokenizer pipeline runs global deduplication as a standard
step, and [Lee et al. 2022](https://aclanthology.org/2022.acl-long.577/)
report that deduplicated models emit memorised text ~10x less often and need
fewer training steps for the same accuracy. For a *tokenizer*
specifically, non-uniform duplication distorts merge frequencies — the exact
quantity the Section 5 heap bug corrupted.

Be precise about what our own control already covers: the upsampled-wiki ×8 run
gave a byte-identical model, which shows invariance to *uniform* scaling and says
nothing about duplication, which is non-uniform by definition.

**Corpus gaps.** Project Madurai (literature/poetry), government and legal open
data, and technical corpora are still unfetched. Larger, and a scope decision
rather than a gap: our corpus has incidental Latin text (markup remnants, proper
nouns) but no *transliterated Tamil* as a represented variety. BharatGen notes
that native scripts dominate formal text while "informal and digital
communication increasingly employ Latin scripts" — so if the target is chat or
social text rather than Wikipedia, this is the biggest untested item here.

### 7.4 Design experiments the literature points at

| Experiment | Source | Why it lands here |
|---|---|---|
| Split digits individually | [BharatGen](https://arxiv.org/html/2508.06533v1); Singh & Strouse 2024 measure ~20% arithmetic-accuracy swings from digit tokenization | our pretokenizer emits digit runs as one chunk (`(?:\p{N}\p{M}*)+`), so `1963` is a single token. Cheap A/B |
| Separate diacritics/matras in pretokenization | BharatGen evaluated exactly this, trading token-to-word ratio against robustness to user diacritic errors | this is the question our syllable clusters answer the opposite way, and the `ும்` pathology — a token starting with a combining mark — arrived by accident, not decision |
| Sandhi-aware tokenization | [Agathiyam](https://openreview.net/forum?id=r3ViBkH9ic) (ICLR 2026 submission, **unrefereed**; the PDF sits behind OpenReview's browser check, so its numbers could not be verified here); [BHARATI](https://arxiv.org/html/2607.23319v1) | Tamil sandhi means whitespace boundaries are not morpheme boundaries, and we only split whitespace |
| Morphology-aware vocabulary | [TamilTok](https://aclanthology.org/2026.dravidianlangtech-1.7/) (DravidianLangTech 2026) puts explicit morpheme structure into tokenizer training and reports gains in alignment *and* downstream performance; MorphTok (Brahma et al. 2025) curated word sets; VerChol grammar-first reaches Tamil fertility 1.86 at 32,991 vocab | TamilTok's **TamilMorph** (480,000+ morphologically segmented word forms) supplies the boundary labels, and [ThamizhiMorph](https://github.com/sarves/thamizhi-morph) (open-source Tamil FST analyser/generator) can supply stems and affixes to seed the base vocab or constrain merges across morpheme boundaries |

### 7.5 Vocabulary size and composition

- The CHiPSAL 2026 comparison (WordPiece vs SentencePiece vs **BBPE**, 5.45 GB
  Tamil corpus, news-title classification) found WordPiece and SentencePiece beat
  BBPE on efficiency *and* accuracy, and that **raising vocabulary size helps
  WordPiece and SentencePiece but not BBPE** — BBPE's zero OOV rate is paid for in
  overfragmentation. Two readings for us: external support for keeping a
  linguistic base vocabulary rather than pure bytes, and a prediction that our
  30K→120K gains keep paying, unlike a pure byte-level model.
- Sarvam-1 reports fertility of **1.4-2.1** across its 10 supported Indic
  languages, framing the problem as **4-8 tokens per word for Indic scripts**
  under general multilingual tokenizers, against 1.4 for English. That 1.4-2.1
  band is the range to be judged against, and it is the same disparity 7.7
  refers to.
- Reserved tokens: BharatGen deliberately leaves many tokens unassigned "to
  accommodate future expansion". We reserve 100. Worth an explicit decision
  rather than an inherited default.
- **Vocabulary utilisation — measured.** `scripts/vocab_utilization.py`, scored
  on both held-out splits (70,000 lines). Utilisation degrades sharply with
  budget, and much faster on the homogeneous corpus:

  | model | never used | used >10x | first 30K merges carry | smallest vocab for 99.9% of emissions |
  |---|---:|---:|---:|---:|
  | wiki 30K | 8.2% | 58.8% | 76.0% | 27,685 |
  | mixed 60K | 7.5% | 36.1% | 72.0% | 56,471 |
  | mixed 120K | 22.9% | 36.1% | 66.8% | 93,214 |
  | wiki 120K | **40.4%** | 54.3% | 66.8% | 71,301 |

  The wiki-only 120K model is the worst of the four — 40% of it does nothing on
  in-domain text, because a homogeneous corpus gives a 120K budget nothing but
  more of the same to overfit. This is an argument *for* the mixed corpus at
  large budgets, independent of the by-domain compression table.

  It does not argue for shrinking 120K. "Unused on 70,000 held-out lines" is not
  "useless" — a large vocab exists partly to cover rare-but-correct input. What
  it does mean is that the 120K recommendation carries ~28,000 tokens that do no
  work in-domain, so its advantage should be read as concentrated in the first
  60–90K.

### 7.6 Harness hygiene

Both defects in Section 5 were found by accident, while running an unrelated
A/B. A third one is cheap to guard against:

- The brute-force oracle runs at 150 merges against the 5 KB sample corpus. Run
  it at several merge counts, alongside the determinism check, in CI rather than
  by hand.
- `spm_ab.py` is invoked manually and reuses unversioned `.model` files under
  `--skip-train`, so a stale reference model would silently change the published
  numbers.

### 7.7 Why this is worth doing at all

The "tokenization tax" literature reports encoding-length differences of **up to
15x** between languages for the same text, with **over 4x** still present in
character-level and byte-level models ([Petrov et al. 2023](https://arxiv.org/abs/2305.15425)),
and the cost framing — API users in many languages are charged more while getting
poorer results ([Ahia et al. 2023](https://arxiv.org/abs/2305.13707)) — treats
this as a latency and access disparity rather than a neutral engineering choice;
[Parity-Aware BPE](https://arxiv.org/abs/2508.04796) makes cross-lingual token
parity its objective outright. Sizing our own position on that gradient is item 4
above — and until it is run, the claim in this paragraph is inherited from other
papers rather than earned here.

Said plainly, because it is a real challenge to this project's premise: the most
recent evidence on the mechanism is **not** supportive. Arnett & Bergen find the
agglutinative performance gap is largely a training-data-size effect, not a
tokenizer effect, and TamilTok's gains come from *training on* morphology rather
than from compression. Both point the same way — the number that matters is a
downstream one, and bytes/token is a proxy not yet shown to carry it in Tamil.
That is the case for item 3 above, not against the tokenizer work.