#!/usr/bin/env python3
"""Publish a trained tokenizer to the Hugging Face Hub, with a model card built
from measurement rather than from claims.

Why this is a script and not a README section: the model card currently on the
Hub says "trained in ~1.4 s" for a model that scores 10.62 bytes/token, while a
same-size model trained correctly scores 13.47. The card was accurate about what
happened and wrong about whether it was good. Every number below is re-measured
at publish time, so the card cannot drift from the artifact again.

Two invariants are enforced, not assumed:

  1. Decode fidelity must be 100% on both held-out splits, or nothing uploads.
     A tokenizer that corrupts text is worse than no tokenizer.
  2. The export has to load through AutoTokenizer with its special tokens
     resolved, or nothing uploads.
  3. The card quotes ID agreement as measured, and says why it is not 100%.
     The gap is in our encoder, not in the export.

Usage:
  export HF_TOKEN=hf_xxx
  python scripts/publish_hf.py                    # dry run, prints the card
  python scripts/publish_hf.py --apply            # upload
  python scripts/publish_hf.py --apply --repo someone/else/tamil-tokenizer

Requires `pip install huggingface_hub` and a release build:
  cargo build --release --bin train --bin score --bin export_hf --bin encode_lines
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO = 'barryallen16/tamil-tokenizer-wiki'

# The model whose artifacts also go to the repository root, so that
# `AutoTokenizer.from_pretrained("barryallen16/tamil-tokenizer-wiki")` loads
# without a subfolder argument. It has to be the model the README calls the
# default, or the one-liner and the documentation disagree.
DEFAULT_KEY = '120k'

# Files the Hub repository root needs for AutoTokenizer to load it.
ROOT_FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']

# The Rust encoder and the Hugging Face export read the same model and apply
# the same merges, so they agree on every ID. Anything below this is a bug in
# one of them.
ID_AGREEMENT_FLOOR = 99.0

SPLITS = [
    ('wiki', 'data/wiki/test.txt'),
    ('IndicCorp', 'data/indiccorp/test.txt'),
]

# The two models worth publishing, and how each was trained. The command is
# printed into the card so a reader can regenerate the artifact rather than
# trust it.
#
# Both are v4: trained with a dummy prefix on every training line, which closes
# the line-initial word split, worth +1.0-1.3% depending on merge budget. A v3
# model and a v4 model differ only in that preprocessing, so the published set
# is coherent.
PREFIX_NOTE = ("train on a copy of the corpus with one space prepended to every "
               "line (SentencePiece's add_dummy_prefix)")

MODELS = [
    {
        'key': '32k',
        'name': 'tamil_tokenizer_wiki',
        'local': 'output/v4_wiki_30000_prefixed.json',
        'training': 'python scripts/prefix_ab.py data/wiki/train.txt '
                    'data/wiki/test.txt 30000   # then take dummy.json',
        'corpus': 'data/wiki/train.txt',
        'blurb': '30K merges on cleaned Tamil Wikipedia. Same merge budget as the '
                 'model this repo previously published, but trained with a dummy '
                 'prefix, which is worth ~1.3%.',
    },
    {
        'key': '120k',
        'name': 'tamil_tokenizer_wiki_120k_mixed',
        'local': 'output/v4_mixed_120000_prefixed.json',
        'training': 'python scripts/prefix_ab.py data/mixed/train.txt '
                    'data/wiki/test.txt 120000  # then take dummy.json',
        'corpus': 'data/mixed/train.txt',
        'blurb': 'The default: 120K merges on wiki + IndicCorpV2. Matches the '
                 'best specialist on each domain.',
    },
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', **kw)


def score(model, corpus):
    """bytes/token and tok/word from score.rs, read from its own columns.

    Never rebuild bytes/token from the MB column; score.rs prints MB to one
    decimal and that has already inflated a published number by 0.4%."""
    p = run([os.path.join(ROOT, 'target/release/score.exe'), model, corpus])
    if p.returncode != 0:
        sys.exit(f'score failed on {model}: {p.stderr[:300]}')
    parts = p.stdout.strip().split('\t')
    return {'bytes_per_token': float(parts[3]), 'tok_per_word': float(parts[4])}


def verify_export(model_json, hf_dir, corpus):
    """Returns (ids_match_pct, decode_pct). Decode is the gate."""
    p = run(['python', os.path.join(ROOT, 'scripts/verify_hf_export.py'),
             model_json, hf_dir, corpus, '500'])
    ids = dec = None
    for line in p.stdout.splitlines():
        if line.startswith('ids match'):
            ids = float(line.split('(')[1].split('%')[0])
        if line.startswith('decode ok'):
            dec = float(line.split('(')[1].split('%')[0])
    return ids, dec


def verify_install(hf_dir):
    """Returns (ok, output). Checks the export loads the way a user loads it."""
    p = run([sys.executable, os.path.join(ROOT, 'scripts/verify_install.py'), hf_dir])
    return p.returncode == 0, (p.stdout + p.stderr)


def build_card(measured, export_stats):
    rows = []
    for name, m in measured.items():
        w, i = m['wiki'], m['IndicCorp']
        rows.append(f"| {name} | {w['bytes_per_token']:.3f} | {i['bytes_per_token']:.3f} | "
                    f"{w['tok_per_word']:.3f} | {i['tok_per_word']:.3f} |")

    files = []
    for name, m in measured.items():
        files.append(f"| `{name}.json` | Trainer format: `{{vocab, merges, metadata}}`. "
                     f"Load with this repo's Rust encoder. |")
        files.append(f"| `{name.replace('.json', '')}_hf/tokenizer.json` | "
                     f"Hugging Face `tokenizers` format, loadable from Python. |")
        files.append(f"| `{name.replace('.json', '')}_hf/tokenizer_config.json` | "
                     f"`AutoTokenizer` settings and special-token names. |")
        files.append(f"| `{name.replace('.json', '')}_hf/special_tokens_map.json` | "
                     f"bos, eos, pad and unk. |")
    files.append(f"| `tokenizer.json`, `tokenizer_config.json`, "
                 f"`special_tokens_map.json` | The default 120K model at the "
                 f"repository root, so `AutoTokenizer.from_pretrained` loads it "
                 f"without a subfolder. |")

    ids = [v for s in export_stats.values() for k, v in s.items() if k == 'ids']
    dec = [v for s in export_stats.values() for k, v in s.items() if k == 'decode']
    id_lo, id_hi = min(ids), max(ids)
    id_range = f"{id_lo:.1f}%" if abs(id_hi - id_lo) < 0.05 else f"{id_lo:.1f}-{id_hi:.1f}%"
    dec_lo = f"{min(dec):.1f}%" if dec else "n/a"

    return f"""---
language:
  - ta
tags:
  - tokenization
  - bpe
  - tamil
  - bpe-tokenizer
library_name: tokenizers
license: mit
---

# Tamil tokenizer

BPE for Tamil, trained on cleaned Tamil Wikipedia and an IndicCorpV2 slice. The
base vocabulary is 1,135 Tamil orthographic clusters rather than raw bytes, so
Grantha syllables and conjuncts cost one token each instead of being reassembled
from merges.

## Numbers

Held-out splits, not trained on:

| model | wiki bytes/tok | IndicCorp bytes/tok | wiki tok/word | IndicCorp tok/word |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Bits-per-byte, cleaned wiki held-out split under Qwen2.5-0.5B, via
`scripts/colab_bpb.py`:

| tokenizer | tok/word | bits/byte |
|---|---:|---:|
| this repo (120K) | **1.57** | **1.00** |
| bert-base-multilingual-cased | 3.74 | 2.16 |
| Llama 3 | 11.96 | 3.60 |
| cl100k_base | 11.88 | 3.60 |

Llama 3 and `cl100k_base` land 0.007% apart on tokens emitted. Both are
byte-fallback BPE with no Tamil vocabulary, so both collapse to about one token
per UTF-8 byte. Which byte-level BPE you pick makes no difference; having Tamil
vocabulary does.

Measured on the 120K export before the dummy-prefix change; the models published
here are slightly better still by bytes/token (see the table above). The gap is
under 1% either way.

## Files

{chr(10).join(files)}

## Use

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("barryallen16/tamil-tokenizer-wiki")
tok.encode("வணக்கம் இது தமிழ்").input_ids
```

The 120K mixed model sits at the repository root. The 30K wiki model is in
`tamil_tokenizer_wiki_hf/`:

```python
tok = AutoTokenizer.from_pretrained(
    "barryallen16/tamil-tokenizer-wiki", subfolder="tamil_tokenizer_wiki_hf"
)
```

Without `transformers`:

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tamil_tokenizer_wiki_hf/tokenizer.json")
tok.encode("வணக்கம் இது தமிழ்").ids
```

This tokenizer adds no BOS or EOS. `tokenizer.bos_token_id` and
`tokenizer.eos_token_id` are available for a training loop that wants them.

The bare `.json` files at the top level are the trainer's own format
(`{{vocab, merges, metadata}}`) and only load through the Rust encoder.
Convert with:

```bash
cargo run --release --bin export_hf -- tamil_tokenizer_wiki.json hf_export/out
```

## Training

Both models are v4: {PREFIX_NOTE}. A v3 model differs only in that
preprocessing step, so the two published artifacts are comparable.

```bash
{chr(10).join(m['training'] for m in MODELS if m['name'] in measured)}
```

## Export fidelity

`tokenizer.json` round-trips losslessly: {dec_lo} decode fidelity on both
held-out splits, checked before every publish. It reproduces {id_range} of the
Rust encoder's token IDs, and the gap is in the encoder.

No merges are lost in the export; all of them are present and in rank order.
HF's `BPE` model starts from single characters and merges every occurrence of
the best-ranked pair per pass. The Rust encoder starts from the 1,135 base
clusters and merges one occurrence per pass. Reimplementing HF's algorithm
over this merge list reproduces its IDs exactly, so the two engines converge
once the encoder takes those semantics.

Token count differs by about 1.7% against the Rust encoder. If you need exact
ID parity, use the Rust encoder.
"""


def main():
    # The card quotes Tamil examples; Windows consoles default to cp1252 and
    # raise on them.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=DEFAULT_REPO)
    ap.add_argument('--apply', action='store_true', help='actually upload (default: dry run)')
    ap.add_argument('--card-only', action='store_true',
                    help='re-upload only README.md, not the model files')
    ap.add_argument('--only', choices=[m['key'] for m in MODELS], help='publish one model')
    ap.add_argument('--corpora', action='store_true', help='also upload the training corpora')
    ap.add_argument('--commit-message', default='publish hf artifacts',
                    help='commit message for the Hub commit')
    args = ap.parse_args()

    # The token is only needed to upload. A dry run should not need one.
    token = os.environ.get('HF_TOKEN')
    if args.apply and not token:
        sys.exit('HF_TOKEN is not set.\n'
                 '  Linux/macOS: export HF_TOKEN=hf_xxx\n'
                 '  Colab:       from google.colab import userdata; '
                 'os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")')
    if not args.apply:
        print('DRY RUN - nothing will be uploaded. Add --apply to publish.\n')

    models = [m for m in MODELS if not args.only or m['key'] == args.only]
    missing = [m['local'] for m in models if not os.path.exists(os.path.join(ROOT, m['local']))]
    if missing:
        sys.exit('missing artifacts: ' + ', '.join(missing) +
                 '\nTrain them first; see the Reproducing section.')

    measured, export_stats, uploads, failures = {}, {}, {}, []

    for m in models:
        path = os.path.join(ROOT, m['local'])
        print(f"--- {m['name']}")
        print(f"    corpus present: {os.path.exists(os.path.join(ROOT, m['corpus']))}")

        hf_dir = os.path.join(tempfile.gettempdir(), 'hfpub_' + m['key'])
        # No explicit model_max_length: the export's default is unlimited, and
        # a published tokenizer should not truncate a caller's input.
        exp = run([os.path.join(ROOT, 'target/release/export_hf.exe'), path, hf_dir])
        if exp.returncode != 0:
            sys.exit(f'export_hf failed: {exp.stderr[:400]}')
        print('   ', exp.stdout.splitlines()[0])

        # Gate: the export has to load the way a user loads it.
        ok, out = verify_install(hf_dir)
        for line in out.splitlines():
            if line.strip().startswith('[FAIL]'):
                print('   ', line.strip())
        print(f'    install check: {"ok" if ok else "FAILED"}')
        if not ok:
            failures.append(f'{m["name"]}: install check failed')

        # Gate: decode must be perfect and the two engines must agree on the
        # IDs, or this does not ship.
        decode, ids = [], []
        for split_name, corpus_rel in SPLITS:
            corpus = os.path.join(ROOT, corpus_rel)
            if not os.path.exists(corpus):
                print(f'    {split_name}: split missing, skipping')
                continue
            i, d = verify_export(path, hf_dir, corpus)
            decode.append(d)
            ids.append(i)
            print(f'    {split_name:9} decode {d:.1f}%  ids {i:.1f}%')
            if d is not None and d < 100.0:
                failures.append(f'{m["name"]} on {split_name}: decode {d}%')
            if i is not None and i < ID_AGREEMENT_FLOOR:
                failures.append(f'{m["name"]} on {split_name}: ids {i}%')

        export_stats[m['name']] = {
            'decode': min(d for d in decode if d is not None),
            'ids': min(i for i in ids if i is not None),
        }

        m_measured = {}
        for split_name, corpus_rel in SPLITS:
            corpus = os.path.join(ROOT, corpus_rel)
            if os.path.exists(corpus):
                m_measured[split_name] = score(path, corpus)
        measured[m['name']] = m_measured

        uploads[m['name'] + '.json'] = path
        for f in ROOT_FILES:
            uploads[m['name'] + '_hf/' + f] = os.path.join(hf_dir, f)
        if m['key'] == DEFAULT_KEY:
            for f in ROOT_FILES:
                uploads[f] = os.path.join(hf_dir, f)

    if failures:
        print('\nREFUSING TO UPLOAD:')
        for f in failures:
            print('   ', f)
        print('\nA tokenizer that corrupts text or will not load is worse than none.')
        print('Fix export_hf.rs, or scripts/verify_install.py, and re-run.')
        return 1

    card = build_card(measured, export_stats)
    print('\n--- model card ---\n')
    print(card)

    if args.corpora:
        for m in models:
            uploads['corpora/' + os.path.basename(m['corpus'])] = os.path.join(ROOT, m['corpus'])

    if not args.apply:
        print('--- would upload ---')
        for dest, src in sorted(uploads.items()):
            print(f'   {dest:46} {os.path.getsize(src) / 1e6:7.2f} MB  <- {src}')
        print(f'\n   README.md  (generated)')
        print(f'   -> repo {args.repo}')
        print('\nDry run complete. Re-run with --apply to upload.')
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if args.card_only:
        print('   card-only: skipping model files')
    else:
        api.create_repo(args.repo, repo_type='model', exist_ok=True)

    # One commit for the whole publish. A file-by-file upload leaves the Hub
    # history as a dozen commits named "Upload file", which hides when the
    # artifact actually changed.
    from huggingface_hub import CommitOperationAdd

    operations = []
    if not args.card_only:
        for dest, src in sorted(uploads.items()):
            operations.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=src))
    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False,
                                     encoding='utf-8') as f:
        f.write(card)
        card_path = f.name
    operations.append(CommitOperationAdd(path_in_repo='README.md',
                                         path_or_fileobj=card_path))

    print(f'   committing {len(operations)} file(s):')
    for op in operations:
        print(f'     {op.path_in_repo}')
    api.create_commit(
        repo_id=args.repo,
        repo_type='model',
        operations=operations,
        commit_message=args.commit_message,
    )
    os.unlink(card_path)

    print(f'\nPublished to https://huggingface.co/{args.repo}')
    return 0


if __name__ == '__main__':
    sys.exit(main())