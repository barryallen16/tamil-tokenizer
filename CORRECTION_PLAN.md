# Tamil Tokenizer — Correction Plan

Five concrete gaps between roadmap claims and shippable reality. Ordered by impact and effort.

---

## RESULTS (measured 2026-10-10)

**Task 2's premise was wrong.** The plan said the export "misses the `Replace`
prepend". It does not — `byte_fallback` and the `<0x00>..<0xFF>` tokens were
already in place, and adding a `Replace` would have *broken* byte fallback
entirely. Han characters already round-tripped. Task 2 below is rewritten to the
bug that actually exists.

**What shipped instead** (`export_hf.rs`): `ByteLevel` removed from the decoder
sequence. `ByteFallback` assembles `<0xNN>` into bytes and returns a UTF-8
string; `ByteLevel` then re-applied the GPT-2 table to the *result*, so a
decoded `é` (U+00E9, itself a GPT-2 stand-in for byte 0xE9) was turned back
into the raw byte 0xE9 and decoded to U+FFFD.

| Lines containing Latin-1 bytes, across 4 corpora | 3,988 |
|---|---:|
| corrupted with `ByteFallback` + `ByteLevel` | **2,074 (52.0%)** |
| corrupted with `ByteFallback` alone | **0** |

This closes roadmap Known Limit #1, which understated it: the defect was not
"HF emits one token where we emit two" but **silent data corruption**.

**Task 1 result — the 99.4–99.7% claim does not reproduce.**

| Model | Corpus | ids match | decode ok |
|---|---|---:|---:|
| v3_mixed_120000 | wiki | 79.2% | **100%** |
| v3_mixed_120000 | IndicCorp | 73.0% | **100%** |
| v2_mixed_30000 | wiki | 79.2% | **100%** |

v2 gives *identical* numbers to v3, so the gap is not a v3 regression — the
roadmap's figure never held for the generation it was attributed to.

**Root cause (confirmed by simulation, not inference).** All 120,000 learned
merges survive the export — zero lost. The divergence is structural: HF's
`BPE` model always starts from individual characters, while `FastBPEEncoder`
starts from the 1,135 base clusters and only then applies merges by rank. Same
merge list, different starting symbols, different segmentations. Replaying both
algorithms in Python over one chunk reproduces both outputs exactly:

| chunk ` வள்ளச்சி` | output |
|---|---|
| base-cluster start (ours) | ` வள்ளச்சி` — 1 token |
| character start (HF) | ` வள்ள` + `ச்சி` — 2 tokens |

Registering base clusters as `added_tokens` was tried and is **worse**: HF
extracts added tokens before the BPE model runs, merges never fire, and output
collapses to characters (0% agreement). HF `BPE` has no seed-symbol concept, so
this export is a lossy approximation of our encoder, not an equivalent.

**Consequence for the roadmap:** hold the export to 100% *decode fidelity* and
stop quoting ~99% ID agreement. Sections 4 and 5's bytes/token figures are
measured against our own encoder and stand — but any HF-hosted consumer of this
tokenizer will see ~20% worse segmentation than the published numbers imply.

**Tasks 3–5: not started.**

---

## 1. Verify HF Export on Recommended Models (v3_*)
**File:** `scripts/verify_hf_export.py` (run), `export_hf.rs` (fix if needed)  
**Time:** 2–3 hours  
**Why:** Roadmap's 99.4–99.7% agreement was measured on `v2_*` models. The recommended default is now `v3_*` (120K mixed) with a completely different merge list. Unverified export = silent ID mismatches for HF users.

### Steps
1. Run export on all three `v3_*` models:
   ```bash
   cargo run --bin export_hf output/v3_wiki_120000.json hf_export/v3_wiki_120000 512
   cargo run --bin export_hf output/v3_mixed_120000.json hf_export/v3_mixed_120000 512
   cargo run --bin export_hf output/v3_corpus_ic_120000.json hf_export/v3_corpus_ic_120000 512
   ```
2. Run `verify_hf_export.py` against both held-out splits (wiki + IndicCorp) for each model.
3. Categorize remainder (as done for v2: OOV, byte-fallback, pretokenizer diff, etc.).
4. If agreement < 99% on any split, debug and fix `export_hf.rs` before publishing.

### Acceptance — measured, criterion was wrong
- ~~≥99% ID agreement~~ — **not achievable, and was never true.** Actual: 79.2%
  (wiki) / 73.0% (IndicCorp), identical on v2 and v3. See RESULTS for root cause.
- **Decode fidelity 100% on both held-out splits — met.**
- Remainder categorized: 1 Latin-1 stand-in line, 0 unvocab, 140 structural.
  The structural category is the base-cluster/character-start difference, not a
  data defect.

---

## 2. Fix HF Byte-Fallback Decoder — DONE, and not the bug this described
**File:** `export_hf.rs`
**Why:** the original text blamed a missing `Replace` prepend. Wrong. Byte
fallback was already working; the bug was `ByteLevel` running *after*
`ByteFallback` and corrupting output.

### Decoder before (corrupts Latin-1):
```json
"decoders": [
  { "type": "ByteFallback" },
  { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true }
]
```

### Decoder after (shipped):
```json
"decoders": [ { "type": "ByteFallback" } ]
```

### Steps — done
1. Drop `ByteLevel` from the decoder sequence in `export_hf.rs`.
2. Re-export all `v3_*` models.
3. Round-trip Latin-1, Han, Tamil, emoji, ASCII: 8/8 pass (was 5/8).
4. `cargo test --release`: 50 passed.
5. Swept 3,988 Latin-1-bearing lines across 4 corpora: 2,074 corrupted → 0.

### Acceptance — met
- Round-trip passes on Han + Tamil + Latin-1 + emoji
- No regression: 50/50 tests pass

---

## 3. Measure Vocabulary Utilization at 120K — DONE
**File:** `scripts/vocab_utilization.py` (new)
**Result:** `v3_mixed_120000`, scored on 70,000 held-out lines (wiki + IndicCorp).

| | count | share |
|---|---:|---:|
| never used | 28,022 | **22.9%** |
| used exactly once | 24,542 | 20.0% |
| used ≤ 10x | 78,254 | 63.9% |
| used > 10x | 44,208 | 36.1% |

Merge rank is the sharper axis — merges are ordered, so rank is meaningful:

| merge ranks | tokens used | share of all emitted tokens |
|---|---:|---:|
| 0–30K | 96.4% | **66.9%** |
| 30K–60K | 88.3% | 7.9% |
| 60K–90K | 71.1% | 3.9% |
| 90K–120K | 55.3% | **2.4%** |

**The last 30K merges — a quarter of the merge budget — carry 2.4% of emitted
tokens.** Smallest vocab retaining 99.9% of emissions: 93,214.

Unused tokens are dominated by markup and Latin residue (`' !!!'`, `' "||'`,
`' >'`, `' BJP'`, `' ASVAB'`), which is the Section 3 markup problem still
visible at 120K.

**Caveat, stated so this is not over-read:** "unused on 70K held-out lines" is
not "useless". A 122K vocab exists partly to cover rare-but-legitimate input
(`' BJP'` may be rare and still correct). This does not argue for shrinking the
vocab; it argues that the 120K recommendation is carrying ~24,000 tokens that do
no work on in-domain text, so the compression gain attributed to 120K should be
re-read as a gain concentrated in the first 60–90K.

### Acceptance — met
- Utilization report printed; decision recorded above. **No change to the 120K
  default**, but the justification is now narrower than the roadmap claims.

---

## 4. Downstream Smoke Test (Bits-per-Byte + Task Accuracy)
**Files:** New training script, small BERT config  
**Time:** 1–2 days (compute-dependent)  
**Why:** Section 7.2 — roadmap admits bytes/token is a flawed proxy. Arnett & Bergen 2025 find agglutinative gap is training-data-size, not tokenizer. A single downstream number validates or falsifies the premise.

### Minimal viable test
1. **Data:** Use existing held-out splits (wiki + IndicCorp) as train/val.
2. **Tokenizers:** 30K wiki, 120K mixed, mBERT (`bert-base-multilingual-cased`), Llama 3 (`meta-llama/Meta-Llama-3-8B` tokenizer).
3. **Model:** Tiny BERT (2 layers, 256 hidden, 4 heads) — trains in ~10 min on GPU.
4. **Task:** Tamil news title classification (CHiPSAL 2026 style) or MLM perplexity.
5. **Metrics:** Bits-per-byte (tokenizer-agnostic), classification accuracy.

### Acceptance
- Table: tokenizer → bits/byte, accuracy, fertility (tok/word)
- If 120K mixed doesn't beat mBERT/Llama on downstream, re-evaluate roadmap premise

---

## 5. Line-Initial Prefix Experiment (New Models Only)
**Files:** `pretokenizer.rs` (optional), training script  
**Time:** 1 day experiment  
**Why:** Known Limit #4 — pretokenizer adds leading space to mid-line chunks only. Cost: ~1% more tokens. Fix for *new* models: prepend dummy prefix to every line before pretokenization (like SentencePiece `add_dummy_prefix`).

### Steps
1. Create training variant that prepends ` ` to every line in `initialize_from_corpus`.
2. Train two 30K models on same corpus: current vs. dummy-prefix.
3. Measure bytes/token on held-out.
4. If gain >0.5%, adopt dummy prefix for all future training (document: invalidates old models).

### Acceptance
- Measured delta recorded; decision: adopt / reject

---

## Execution Order

| Phase | Tasks | Dependency |
|---|---|---|
| **Immediate** | 1, 2 | None — export fixes unblock publishing |
| **Quick Win** | 3 | None — 30 min script |
| **Validation** | 4 | Needs GPU; can run in parallel |
| **Future-Proof** | 5 | Only if publishing new models |

---

## Files to Modify

| File | Change |
|---|---|
| `export_hf.rs` | **done** — dropped `ByteLevel` from decoder sequence; header rewritten |
| `scripts/verify_hf_export.py` | **done** — removed the wrong `Replace` advice, added expected-value baseline |
| `.gitignore` | **done** — ignore generated `hf_export/` |
| `scripts/vocab_utilization.py` | new script (Task 3, not started) |
| `trainer.rs` / training script | Dummy prefix variant (Task 5, not started) |

---

## Non-Goals (Explicitly Deferred)

- Tamil agglutination/affix handling (Section 3) — needs downstream validation first
- Unigram trainer — correctly rejected (0.6% gain)
- Corpus deduplication — needs downstream validation first
- MorphScore / TokEval integration — needs downstream validation first

---

## Handoff Notes for Next Agent

1. **Start with Task 1 + 2** — they unblock HF publication and fix a known user-facing bug.
2. **Task 3** is a standalone script — can run anytime.
3. **Task 4** requires GPU — if unavailable, run bits-per-byte only (no training) using `score.rs` on a small LM checkpoint.
4. **Task 5** only matters if you plan to train *new* models after this.
5. All scripts should write results to `output/correction_results/` for traceability.
6. Run `cargo test` after any `export_hf.rs` or `trainer.rs` changes.