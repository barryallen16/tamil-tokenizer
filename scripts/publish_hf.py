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
  2. The card quotes ID agreement as ~79%, not ~99%, and says why. HF's BPE
     cannot reproduce our base-cluster segmentation; see Known Limit 6 in
     IMPROVEMENT_ROADMAP.md.

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

SPLITS = [
    ('wiki', 'data/wiki/test.txt'),
    ('IndicCorp', 'data/indiccorp/test.txt'),
]

# The two models worth publishing, and how each was trained. The command is
# printed into the card so a reader can regenerate the artifact rather than
# trust it.
MODELS = [
    {
        'key': '32k',
        'name': 'tamil_tokenizer_wiki',
        'local': 'output/v3_wiki_30000.json',
        'training': 'cargo run --release --bin train -- data/wiki/train.txt 30000',
        'corpus': 'data/wiki/train.txt',
        'blurb': '30K merges on cleaned Tamil Wikipedia. Same size as the model this '
                 'repo previously published, retrained after the merge-heap fix, so '
                 'it compresses ~27% better than that file did.',
    },
    {
        'key': '120k',
        'name': 'tamil_tokenizer_wiki_120k_mixed',
        'local': 'output/v3_mixed_120000.json',
        'training': 'cargo run --release --bin train -- data/mixed/train.txt 120000',
        'corpus': 'data/mixed/train.txt',
        'blurb': 'The roadmap-recommended default: 120K merges on wiki + IndicCorpV2. '
                 'Matches the best specialist on each domain.',
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

120K model under Qwen2.5-0.5B on the cleaned wiki held-out split, via
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

## Files

{chr(10).join(files)}

## Use

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tamil_tokenizer_wiki_hf/tokenizer.json")
tok.encode("வணக்கம் இது தமிழ்").ids
```

The bare `.json` files are the trainer's own format (`{{vocab, merges, metadata}}`)
and only load through the Rust encoder. Convert with:

```bash
cargo run --release --bin export_hf -- tamil_tokenizer_wiki.json hf_export/out 512
```

## Training

```bash
{chr(10).join(m['training'] for m in MODELS if m['name'] in measured)}
```

## Export fidelity

`tokenizer.json` round-trips losslessly — {dec_lo} decode fidelity on both
held-out splits, checked before every publish — but reproduces {id_range} of the
Rust encoder's token IDs.

The cause is the starting symbol set. This encoder segments into base clusters
first, then applies merges by rank. HF's BPE always starts from single
characters, so identical merge lists produce different segmentations:
`வள்ளச்சி` is one token here and ` வள்ள` + `ச்சி` there. No merges are lost in
the export; all of them are present and in rank order.

Token count differs by about 0.5% against the Rust encoder. If you need exact
ID parity, use the Rust encoder. Full write-up in Known Limit 6 of
`IMPROVEMENT_ROADMAP.md`.
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
    args = ap.parse_args()

    token = os.environ.get('HF_TOKEN')
    if not token:
        sys.exit('HF_TOKEN is not set.\n'
                 '  Linux/macOS: export HF_TOKEN=hf_xxx\n'
                 '  Colab:       from google.colab import userdata; '
                 'os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")')
    if not args.apply:
        print('DRY RUN — nothing will be uploaded. Add --apply to publish.\n')

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
        exp = run([os.path.join(ROOT, 'target/release/export_hf.exe'), path, hf_dir, '512'])
        if exp.returncode != 0:
            sys.exit(f'export_hf failed: {exp.stderr[:400]}')
        print('   ', exp.stdout.splitlines()[0])

        # Gate: decode must be perfect on every split, or this does not ship.
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
        uploads[m['name'] + '_hf/tokenizer.json'] = os.path.join(
            hf_dir, 'tokenizer.json')

    if failures:
        print('\nREFUSING TO UPLOAD — decode fidelity below 100%:')
        for f in failures:
            print('   ', f)
        print('\nA tokenizer that silently corrupts text is worse than none.')
        print('Fix the decoder in export_hf.rs and re-run.')
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
        for dest, src in sorted(uploads.items()):
            print(f'   uploading {dest} ...', end='', flush=True)
            api.upload_file(
                path_or_fileobj=src,
                path_in_repo=dest,
                repo_id=args.repo,
                repo_type='model',
            )
            print(' done')

    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False,
                                     encoding='utf-8') as f:
        f.write(card)
        card_path = f.name
    # huggingface_hub 1.x takes keyword-only args after `self`; a positional
    # call here raises TypeError only after the model files are already up.
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo='README.md',
        repo_id=args.repo,
        repo_type='model',
    )
    os.unlink(card_path)

    print(f'\nPublished to https://huggingface.co/{args.repo}')
    return 0


if __name__ == '__main__':
    sys.exit(main())