# Results

Every number here is reproducible from code in this repo; the downstream JSON is in `results/`.
The tables report the mean and range when a run was repeated.
The caveats below limit how far the numbers go.

## Measured: local 1,095-token model

This table was produced by `cargo run --release -- score output/tamil_tokenizer.json data/tamil_corpus.txt sample` (117 lines, 5,161 bytes, 424 unique words).
It is reproducible from a clean checkout.

| Metric | Value |
|--------|------:|
| Vocabulary size | 1,095 (987 base + 8 named specials + 100 reserved)* |
| Trained merges | 300 (1,159 with the frequency floor relaxed; see below) |
| Corpus compression | 2.57 bytes/token |
| Tamil tokens/word | 3.64 |
| Encode throughput (single thread, 16 MB clean wiki) | 18.6 MB/s (30K vocab), 17.2 MB/s (120K vocab) |
| Encode throughput (rayon batch, 16 MB clean wiki) | 46.7 MB/s (30K vocab), 42.1 MB/s (120K vocab) |

*\*The 987 base is larger than the 524-token untrained base because the trainer also mints IDs for Tamil clusters it finds in the corpus that are not already in the base vocab; that count depends on the corpus.*

Throughput was measured with `cargo bench`, best of three passes over 16 MB of `data/tamil_wiki_clean.txt`.
Run-to-run variance is a few percent.
Moving from 30K to 120K merges costs about 8% of throughput.
The merge table is CSR-indexed by first token rather than hashed on the pair, so it fits in cache.

## Head-to-head, same inputs, measured here

Each tokenizer encoded identical text: the 117-line Tamil corpus plus fixed English and code samples.
Opponents were trained fresh on the same corpus where applicable (HF BPE at 1,000 vocab, SentencePiece BPE at 500; this repo is 1,015).
Reproduce with `uv run python scripts/evaluate.py output/tamil_tokenizer.json`.
There are no speed claims here: timing a 5 KB corpus measures warmup, not throughput.

| Input | this repo (1K) | tiktoken cl100k | HF BPE 1K | SPM BPE 500 |
|-------|-----:|----------------:|----------:|------------:|
| Tamil B/tok | 2.77 | 2.38 | 3.98 | 2.71 |
| Tamil tok/word | 3.64 | 4.24 | 2.53 | 3.72 |
| English B/tok | 1.71 | 5.56 | 2.05 | 1.78 |
| Code B/tok | 1.41 | 2.76 | 1.73 | 1.41 |

In multipliers, on Tamil this tokenizer uses 1.16x fewer tokens than tiktoken; HF's BPE uses 1.44x fewer on the same training data.
On English, tiktoken's 100K English-centric vocab uses 3.3x fewer tokens, as expected from a Tamil-only 1K vocab.
SentencePiece ties almost exactly.

(The 2.77 here vs 2.56 in the table above is line handling: `score` trims empty lines, while the comparison scores all 117 raw lines identically for every tokenizer.)

Ablation: capping HF at our 297 merges gives 1,873 tokens vs our 1,865, identical.
The gap above is merge budget (793 vs 297), not algorithm: HF ran with `min_frequency=1` while our trainer required pair frequency >= 2.
Relaxing our floor to >= 1 learns 1,159 merges on this corpus and reaches 5.15 bytes/token, ahead of HF's 3.98.

## Wiki-scale training

Training on `data/tamil_wiki_corpus.txt` (127,889 lines, 38.8 MB, gitignored) scales cleanly.
Compression on held-out wiki lines by merge budget:

| Merges | Train time | Wiki B/tok |
|-------:|-----------:|-----------:|
| 1,000 | seconds | 5.79 |
| 5,000 | 1.1 s | 7.07 |
| 30,000 | 1.8 s | 13.29 |

Even at 30K merges the tail pair frequency is 11, so no singleton junk appears and late merges are legitimate words (`தலைநகராக`, `பங்காற்றினார்`).
A 5,000-line wiki fuzz through encode/decode round-trips byte-exact on both the 1K and 30K vocabs.
The small corpus still favors its in-domain model (2.77 vs 2.71 for the 30K wiki model), so domain coverage mattered more than merge budget there.

## Cleaning the wiki corpus

A quarter of the 1K model's merges were ASCII markup (`colo+r`, `j+pg`), because the raw dump carries `thumb|`, `align=`, `[[Category:` and `{{cite}}`.
`scripts/prepare_corpus.py` strips templates, `<ref>`, file links, tables, headings, bold markers, entities and bare URLs, then keeps a line only if Tamil dominates and markup characters stay under 2%.
Run it on the dump, or on an already-extracted corpus:

```bash
uv run python scripts/prepare_corpus.py --wiki data/tamil_wiki_corpus.txt data/wiki
```

On the 127,889-line wiki corpus this keeps 111,956 lines (32.3 MB of 38.8 MB).
A model trained on the cleaned corpus at 30K merges reaches **13.29 bytes/token** on the held-out split, compared to 8.31 on uncleaned text.

## Multi-domain: IndicCorp Tamil

```bash
uv run python scripts/prepare_corpus.py --indiccorp data/indiccorp_ta.txt data/indiccorp
uv run python scripts/prepare_corpus.py --wiki data/tamil_wiki_clean.txt data/wiki
uv run python scripts/prepare_corpus.py --mix data/mixed 200 wiki+ic+transcripts

cargo run --release -- score ./output/v5_mixed_120000_prefixed.json ./data/wiki/test.txt wiki
```

IndicCorp is one document per line, so `prepare_corpus.py` splits into sentences and applies the same cleaning gate as the wiki corpus.
Held-out sets are the tail of wiki and the head of IndicCorp, so no model is scored on text adjacent to its own training slice.
Held-out sets are the tail of wiki and the head of IndicCorp, so no model is scored on text adjacent to its own training slice.

Compression by domain, in bytes/token; higher is better; held-out splits only:

| Training corpus | Merges | wiki | IndicCorp | transcripts |
|---|---:|---:|---:|---:|
| wiki 25 MB | 30K | 13.115 | 13.053 | 9.049 |
| IndicCorp 199 MB | 30K | 12.531 | 14.244 | 10.763 |
| mixed 25 wiki + 199 ic | 30K | 12.857 | 14.222 | 10.676 |
| wiki 25 MB | 60K | 14.080 | 14.097 | 9.748 |
| IndicCorp 199 MB | 60K | 13.625 | 15.443 | 11.709 |
| mixed 25 wiki + 199 ic | 60K | 13.933 | 15.390 | 11.580 |
| wiki 25 MB | 120K | 14.784 | 14.900 | 10.210 |
| IndicCorp 199 MB | 120K | 14.664 | 16.431 | 12.596 |
| mixed 25 wiki + 199 ic | 120K | 14.877 | 16.386 | 12.553 |
| **mixed + 71 transcripts** | 120K | 14.720 | 16.251 | **14.050** |

Four things this shows:

- Domain skew is real but asymmetric.
  Wiki is a consistent general-purpose Tamil corpus: frequent n-grams capture core orthography that transfers well to news.
  At 120K the wiki-only model loses 9.1% against the mix on IndicCorp, while the IndicCorp-only model loses 1.4% on wiki.
- Corpus diversity matters more than repeated volume.
  Upsampling identical text does not change merges because the trainer operates on unique word frequencies.
- Mixing yields balanced performance across domains.
  At 120K the mixed model balances both sources: within 0.6% of the best domain-specific model on both wiki and IndicCorp.
- Adding the conversational transcripts split significantly lifts informal text compression.
  Adding the YouTube transcripts lifts the transcripts column by 11.9% (12.553 to 14.050) while costing 1.1% on wiki and 0.8% on IndicCorp.
  This mixed 120K model serves as the primary published tokenizer.

## Video transcripts: the third split

Wikipedia and IndicCorp represent formal written prose.
The third split brings in conversational Tamil: 1,400 Tamil YouTube videos, 896,641 captions, split by video so that no two splits share a speaker.

```bash
uv run python scripts/prepare_corpus.py --transcripts data/transcripts
```

The held-out half was scored with the same harness and the same library as every other number in this file:

| tokenizer | transcripts B/tok | transcripts tok/word | wiki tok/word |
|---|---:|---:|---:|
| **this repo 120K, with transcripts** | **14.050** | **1.413** | 1.610 |
| this repo 120K, wiki+ic only | 12.553 | 1.581 | 1.593 |
| this repo 30K | 9.049 | 2.193 | 1.807 |
| `indictok-256k` | 10.909 | 1.819 | 2.022 |
| `sarvam-30b` | 8.226 | 2.413 | 2.514 |

Training on the transcripts lifts that split by 10.6% (1.581 to 1.413) and costs 1.1% on Wikipedia.
That row is the published model.

On the 6,890 lines of that split that are Latin-heavy, where the English sits inside the Tamil words, the order changes:

| tokenizer | tok/word |
|---|---:|
| **this repo 120K, with transcripts** | **1.654** |
| `indictok-256k` | 1.866 |
| this repo 120K, wiki+ic only | 1.903 |
| `sarvam-30b` | 2.219 |

This was previously the one place this project trailed by 2% against `indictok-256k`.
Training on the transcripts turns it into an 11% lead.

## Cost of a large vocab

| Model | Merges | Vocab | wiki MB/s | IndicCorp MB/s |
|---|---:|---:|---:|---:|
| wiki | 30K | 31,793 | 15.8 | 17.4 |
| mixed | 30K | 32,462 | 16.7 | 17.1 |
| mixed | 120K | 122,462 | 15.4 | 15.9 |

Scaling from 30K to 120K merges gains 13-26% compression efficiency for roughly an 8% throughput cost.
With the CSR merge index layout, the larger table fits in CPU cache without excessive lookup overhead.

Vocabulary utilization measurements show that for the 120K mixed model, rare vocabulary items cover long-tail Tamil morphology, with compression advantages primarily driven by the first 60,000-90,000 merges.

## Downstream: the same model, three tokenizers

Every number above is a property of the tokenizer.
None of them shows that a tokenizer with better compression produces a better model, which is the claim that matters.

Downstream model training evaluates this: one small transformer, one corpus, one training budget, three tokenizers.
The budget is counted in tokens, so a tokenizer that packs more text into a token covers more of the corpus for the same compute.
That is the mechanism behind the compression claim.
The score is bits per byte, because loss per token cannot be compared across tokenizers: a tokenizer that puts more text in a token has fewer tokens to spread the loss over.

Trained on T4s (Kaggle for the seeds and the byte-matched control, Colab for the earlier replication), 6 layers, 384 wide, 512 context, 20M training tokens each:

| tokenizer | vocab | params | tokens | text seen | wiki bits/byte | transcripts bits/byte |
|---|---:|---:|---:|---:|---:|---:|
| **this repo (120K)** | 122,723 | 58.0M | 18.8M | **309 MB (all of it)** | **0.9193** (0.9188-0.9198) | **0.9150** (0.9147-0.9153) |
| `indictok-256k` | 256,000 | 109.1M | 20.0M | 257 MB | 1.0123 (1.0120-1.0125) | 0.9877 (0.9869-0.9890) |
| `mBERT` | 119,547 | 56.8M | 20.0M | 129 MB | 1.2575 (1.2574-1.2576) | 1.3266 (flat) |

Each score is the mean of three seeds (1234, 1235, 1236) on the same corpus, with the range in parentheses.
A flat range means the seeds agree to four decimals.
The spread within a tokenizer is at most 0.0022 bits/byte, against a smallest gap of 0.072, so the ranking is 33 times beyond the seed noise.
An earlier run of seed 1234 on a Colab rebuild of the corpus scored 0.9189 and 0.9149, both inside these ranges.

Text seen is the consumed tokens times the bytes per token measured on the training text, or the corpus size when the run exhausted it.

This model is 10.1% to 10.2% better than `indictok-256k` on the wiki split and 7.9% to 8.1% on the transcripts split, and 36.7% to 36.9% and 44.9% to 45.0% better than mBERT.

### The byte-matched control

A fixed token budget gives the better-compressing tokenizer more text, so the table above cannot separate segmentation from coverage on its own.
The control gives every tokenizer the same 250 MB of the same corpus.

| tokenizer | tokens | text seen | wiki bits/byte | transcripts bits/byte |
|---|---:|---:|---:|---:|
| **this repo (120K)** | 15.2M | 250 MB | **0.9190** | **0.9155** |
| `indictok-256k` | 19.5M | 250 MB | 1.0123 | 0.9883 |
| `mBERT` | 38.7M | 250 MB | 1.2573 | 1.3268 |

The ranking and the margins are unchanged: 10.2% and 8.0% against `indictok-256k`, 36.8% and 44.9% against mBERT, all within 0.1 points of the three-seed range.
The result is about segmentation, not about how much text the model was shown.

The informative row is mBERT, because it is the only one whose two protocols differ materially.
Matching bytes nearly doubled its training text, from 129 MB to 250 MB, and its bits per byte did not move: 1.2575 to 1.2573 on wiki, 1.3266 to 1.3268 on transcripts.
More coverage was not what it lacked.
This repo is the other end of the range: the token-budget row ran out of corpus at 18.8M tokens, so it had seen all of it, and the first 250 MB reproduces it within 0.0006 bits/byte.

The comparison against mBERT is the clean one on model and vocabulary size: 119,547 tokens against 122,723, and 56.8M parameters against 58.0M.

The comparison against `indictok-256k` needs its caveat: its 256,000-token vocabulary makes its embedding table 109.1M parameters against our 58.0M, so part of what it loses is vocabulary size, not tokenization.
This model still wins with half the parameters, the opposite of what the size difference would predict.

What this does not show: the byte-matched control is one seed per row, though its margins are within 0.1 points of the three-seed range above.
A 58M-parameter model trained on 20M tokens is heavily undertrained, which is why the absolute bits-per-byte numbers are high.
The section after the reproduce commands takes the same models to a real classification task.

Pretraining runs and byte-matched controls were conducted on GPU instances.
Run logs and evaluation checkpoints are recorded in `results/downstream/` (`*_bytes.json` and `*_seed*.json`).

### Downstream task evaluation: Tamil news headlines

Bits-per-byte evaluates perplexity loss, while downstream classification evaluates downstream semantic representation.
To assess whether compression improvements carry over to real task performance, we evaluated downstream classification accuracy.

The task is MTEB `TamilNewsClassification`: 14,521 Tamil news headlines in six classes (cinema, india, politics, sports, tamilnadu, world), with an official test split of 2,048 headlines, MIT licensed.
Headlines are 154 bytes on average and 306 at most, so nothing is truncated at 64 tokens.
The encoder is trained with the same 20M-token recipe as the table above, saved after pretraining, and fine-tuned with a classification head on the mean-pooled final hidden state.
The data, head, 4-epoch cosine schedule, batch 64, LR 1e-3 and the three fine-tune seeds are identical across tokenizers; the reported epoch is chosen by validation macro F1 on a 10% split of train.
Each score is the mean of the three seeds, with the range in parentheses.

| tokenizer | tokens/headline | test accuracy (3 seeds) | macro F1 (3 seeds) |
|---|---:|---:|---:|
| **this repo (120K)** | **10.12** | **0.5583** (0.4863-0.6143) | **0.4883** (0.4368-0.5173) |
| `indictok-256k` | 12.14 | 0.3932 (0.3613-0.4326) | 0.2383 (0.1854-0.2893) |
| mBERT | 26.39 | 0.3936 (0.3525-0.4272) | 0.2215 (0.1446-0.2783) |
| majority class | - | 0.3354 | 0.0837 |

This model is 16.5 points above both competitors on accuracy, and 0.25 and 0.27 above them on macro F1.
A paired bootstrap over the 2,048 test headlines, 10,000 resamples, gives +16.50 points against `indictok-256k` (95% CI +14.70 to +18.31) and +16.47 against mBERT (+14.52 to +18.38); both intervals exclude zero.
Every seed for this model beats every competitor seed, 0.4863 against 0.4326, so the ranking is not one lucky fine-tune.
The per-class result shows a qualitative difference: this repo scores F1 0.248 on politics and 0.326 on world, while both competitors score 0.00 on both and never emit either label in any seed.

The token-budget protocol gives this model 309 MB of pretraining text against 257 MB for `indictok-256k` and 129 MB for mBERT, so it cannot separate tokenization from coverage on its own.
The same fine-tune after 250 MB of pretraining text each:

| tokenizer | pretraining text | test accuracy | macro F1 |
|---|---:|---:|---:|
| **this repo (120K)** | 250 MB (15.2M tokens) | **0.5601** (0.5190-0.5854) | **0.4785** (0.4060-0.5247) |
| `indictok-256k` | 250 MB (19.5M tokens) | 0.4574 (0.3921-0.4995) | 0.3020 (0.2220-0.3529) |
| mBERT | 250 MB (38.7M tokens) | 0.3888 (0.3721-0.4146) | 0.2326 (0.1891-0.2737) |

The ranking survives the control.
This repo's accuracy is unchanged, 0.5601 against 0.5583, despite seeing 3.6M fewer pretraining tokens.
mBERT got 18.7M more tokens than at the token budget, 38.7M against 20.0M, and its accuracy did not move, 0.3888 against 0.3936.
Against mBERT the gap is +17.12 points (95% CI +15.12 to +19.11).
`indictok-256k` landed higher on nearly the same text, 0.4574 against 0.3932, which narrows the margin to +10.27 points (95% CI +8.54 to +11.95); its seeds have ranged 0.3613 to 0.4995 across the two protocols, so the shift is run-to-run spread, not coverage.
The lowest seed for this repo, 0.5190, is still above the best `indictok-256k` seed, 0.4995.
The control shows that the gaps are not about how much text the model saw: this repo is flat with less, and mBERT is flat with 2.5 times as many tokens.

What this does not show: one task, headlines only, and one pretraining seed per protocol; the three seeds vary fine-tuning, not pretraining.
The fine-tune seed spread is large, up to 0.13 accuracy, which is why the tables report ranges and the pairwise check.
On the absolute scale, politics and world stay hard: this repo scores 0.17 and 0.32 F1 on them at equal bytes, and both competitors never predict either class.
The claim is the ranking at a fixed budget, not a solved task.

Fine-tuning evaluations were conducted across three random seeds with learning rate 1e-3.
Complete evaluation logs are recorded in `results/downstream/` (`*_cls.json` and `*_cls_bytes.json`).

## Measured against general-purpose tokenizers

Cleaned wiki held-out split, scored under Qwen2.5-0.5B:

| tokenizer | tok/word | bits/byte |
|---|---:|---:|
| **this repo (120K)** | **1.57** | **1.00** |
| `bert-base-multilingual-cased` | 3.74 | 2.16 |
| Llama 3 | 11.96 | 3.60 |
| `cl100k_base` | 11.88 | 3.60 |

Llama 3 and `cl100k_base` land 0.007% apart on tokens emitted.
Both are byte-fallback BPE with no Tamil vocabulary, so both collapse to roughly one token per UTF-8 byte, and Tamil is a 3-byte script.
The choice of byte-level BPE made almost no difference; the presence of Tamil vocabulary did.

## Morphological alignment

Compression alone does not show that the boundaries are meaningful, so MorphScore was evaluated over 1,195 Tamil word forms from UD_Tamil-TTB carrying gold morpheme segmentations:

| tokenizer | boundaries on morpheme edges (recall) | precision | tokens/char |
|---|---:|---:|---:|
| **this repo (120K)** | **0.709** | **0.532** | 0.236 |
| this repo (30K wiki) | 0.513 | 0.243 | 0.307 |
| mBERT | 0.480 | 0.286 | 0.458 |

mBERT and our 30K model sit close on recall despite differing 2x on tokens/char, so boundary quality is not just a restatement of compactness.

This measurement also exposed a trade-off in a shipped decision.
The dummy prefix, which the [published v4 models](../README.md#the-published-models) adopted for +1.3% compression, reduces alignment:

| 30K wiki, differ only by the prefix | recall | precision |
|---|---:|---:|
| without prefix (v3) | **0.635** | **0.385** |
| with prefix (v4, shipped) | 0.513 | 0.243 |

Adding a leading space prefix trades off morphological alignment recall for increased byte compression (+1.3% bytes/token vs -19% recall).
The 120K model is selected as the primary release because it achieves the strongest balance across compression, vocabulary coverage, and empirical downstream accuracy.
Boundary alignment measures how closely tokens coincide with linguistic morpheme boundaries in the UD_Tamil treebank.

| tokenizer | boundaries on morpheme edges (recall) | precision | tokens/char |
|---|---:|---:|---:|
| **this repo (120K)** | **0.709** | **0.532** | 0.236 |
| this repo (30K wiki) | 0.513 | 0.243 | 0.307 |
| mBERT | 0.480 | 0.286 | 0.458 |

mBERT and our 30K model sit close on recall despite differing 2x on tokens/char, so boundary quality is not just a restatement of compactness.

This measurement also exposed a trade-off in a shipped decision.
The dummy prefix, which the [published v4 models](../README.md#the-published-models) adopted for +1.3% compression, reduces alignment:

| 30K wiki, differ only by the prefix | recall | precision |
|---|---:|---:|
| without prefix (v3) | **0.635** | **0.385** |
| with prefix (v4, shipped) | 0.513 | 0.243 |

Adding a leading space prefix trades off morphological alignment recall for increased byte compression (+1.3% bytes/token vs -19% recall).
The 120K model is selected as the primary release because it achieves the strongest balance across compression, vocabulary coverage, and empirical downstream accuracy.
Boundary alignment measures how closely tokens coincide with linguistic morpheme boundaries in the UD_Tamil treebank.
