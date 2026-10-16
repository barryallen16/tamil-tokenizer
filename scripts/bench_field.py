#!/usr/bin/env python3
"""Score this tokenizer against the field on one corpus, with the same code path.

Every candidate encodes the same bytes with the same library, so the numbers
compare tokenizers rather than harnesses. The word count is whitespace-split
words, the same rule the rest of this project uses.

Candidates can be a local `tokenizer.json` or a Hub repo id. A repo id is
downloaded to the cache on first use.

Usage:
  python scripts/bench_field.py <corpus.txt> [candidate ...]

  python scripts/bench_field.py data/transcripts/test.txt \
      output/v4_mixed_120000_prefixed.json \
      theschoolofai/BrahmicTokenizer-131K \
      vivekvar/indictok-256k \
      sarvamai/sarvam-30b

  python scripts/bench_field.py --latin-only data/transcripts/test.txt

`--latin-only` keeps the lines that are at least 15% Latin letters, which is
where English words sit inside Tamil sentences.

Requires `pip install tokenizers`.
"""
import argparse
import os
import re
import sys

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LATIN = re.compile(r'[A-Za-z]')

DEFAULT_CANDIDATES = [
    ('ours 120K', 'output/v4_mixed_120000_prefixed.json'),
    ('ours 30K', 'output/v4_wiki_30000_prefixed.json'),
    ('Brahmic-131K', 'theschoolofai/BrahmicTokenizer-131K'),
    ('indictok-256k', 'vivekvar/indictok-256k'),
    ('sarvam-30b', 'sarvamai/sarvam-30b'),
]


def load(name, path):
    """A local file, or a Hub repo id.

    A file in this repo's trainer format is exported to the Hugging Face
    format first, so every candidate goes through the same library and the
    same code path. Comparing our Rust encoder against a Python tokenizer
    would compare the two implementations as much as the two models.
    """
    if path.endswith('.json') or os.path.exists(path):
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f'  {name}: no such file: {full}', file=sys.stderr)
            return None
        if _is_trainer_format(full):
            exported = _export(full)
            if exported is None:
                return None
            full = exported
        return Tokenizer.from_file(full)
    try:
        return Tokenizer.from_pretrained(path)
    except Exception as e:
        print(f'  {name}: could not load {path}: {type(e).__name__}', file=sys.stderr)
        return None


def _is_trainer_format(path):
    try:
        with open(path, encoding='utf-8') as f:
            head = f.read(4096)
    except OSError:
        return False
    return '"vocab"' in head and '"model"' not in head


def _export(path):
    """Run export_hf and return the tokenizer.json it writes."""
    import subprocess
    import tempfile

    binary = os.path.join(ROOT, 'target', 'release',
                          'export_hf.exe' if os.name == 'nt' else 'export_hf')
    if not os.path.exists(binary):
        print(f'  {path}: needs a release build of export_hf', file=sys.stderr)
        return None
    out = tempfile.mkdtemp(prefix='benchfield_')
    proc = subprocess.run([binary, path, out], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'  {path}: export_hf failed: {proc.stderr[:200]}', file=sys.stderr)
        return None
    return os.path.join(out, 'tokenizer.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus')
    ap.add_argument('candidates', nargs='*')
    ap.add_argument('--latin-only', action='store_true',
                    help='keep only the lines that are at least 15% Latin letters')
    args = ap.parse_args()

    corpus = args.corpus
    candidates = args.candidates
    if not candidates:
        candidates = DEFAULT_CANDIDATES

    with open(corpus, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    if args.latin_only:
        before = len(lines)
        lines = [l for l in lines if len(LATIN.findall(l)) >= max(3, 0.15 * len(l))]
        print(f'latin-only: {len(lines)} of {before} lines')

    nbytes = sum(len(l.encode('utf-8')) for l in lines)
    nwords = sum(len(l.split()) for l in lines)
    print(f'corpus={corpus} lines={len(lines)} bytes={nbytes} words={nwords}')
    print(f'{"tokenizer":22} {"tokens":>10} {"B/tok":>8} {"tok/word":>9}')

    for spec in candidates:
        label, _, path = spec.partition('=')
        if not path:
            label, path = os.path.basename(spec), spec
        tok = load(label, path)
        if tok is None:
            print(f'{label:22} {"LOAD FAILED":>10}')
            continue
        count = 0
        for line in lines:
            count += len(tok.encode(line, add_special_tokens=False).ids)
        print(f'{label:22} {count:>10} {nbytes / count:>8.3f} {count / nwords:>9.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
