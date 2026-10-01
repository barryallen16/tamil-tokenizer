#!/usr/bin/env python3
"""A/B the line-initial prefix problem. See roadmap Section 1 and Known Limit 4.

The pretokenizer attaches one leading space to a mid-line chunk (` கணிதம்`)
but not to the first chunk of a line (`கணிதம்`). Those are different symbol
sequences with their own pair statistics, so only the spaced form earns merges
and a line-initial word costs extra tokens.

SentencePiece fixes this with `add_dummy_prefix`. The equivalent here needs no
code change at all: prepend one space to every *training* line, so the trainer
sees only the spaced form, then score on the *unprefixed* held-out split, which
is what real input looks like.

Usage:
  python scripts/prefix_ab.py <train.txt> <heldout.txt> [merges] [workdir]

Trains two models on the same corpus and prints bytes/token for both.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_BIN = os.path.join(ROOT, 'target', 'release', 'train.exe')
SCORE_BIN = os.path.join(ROOT, 'target', 'release', 'score.exe')


def prefixed(src, dst):
    """One space on every line. Written as bytes: the corpus is UTF-8 and a
    decode/encode round-trip here would be a chance to corrupt it."""
    n = 0
    with open(src, 'rb') as f, open(dst, 'wb') as g:
        for line in f:
            g.write(b' ' + line if line.strip() else line)
            n += 1
    return n


def train(corpus, merges, out):
    subprocess.run([TRAIN_BIN, corpus, str(merges), out], check=True,
                   stdout=subprocess.DEVNULL)
    return out


def score(model, corpus):
    """Returns bytes/token from score.rs. Its stdout is a tab-separated row;
    column 3 is bytes/token. Read from the file, not from the MB column:
    score.rs prints MB to one decimal, which the roadmap already flagged as
    having inflated a published number by 0.4%."""
    out = subprocess.run([SCORE_BIN, model, corpus], check=True,
                         capture_output=True, text=True, encoding='utf-8').stdout
    return float(out.strip().split('\t')[3])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    train_txt, heldout = sys.argv[1], sys.argv[2]
    merges = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    workdir = sys.argv[4] if len(sys.argv) > 4 else tempfile.mkdtemp(prefix='prefix_ab_')
    os.makedirs(workdir, exist_ok=True)

    for b in (TRAIN_BIN, SCORE_BIN):
        if not os.path.exists(b):
            sys.exit(f'missing {b}\n  cargo build --release --bin train --bin score')

    print(f'train      {train_txt}')
    print(f'held-out   {heldout}  (never prefixed)')
    print(f'merges     {merges}')
    print(f'workdir    {workdir}\n')

    pref_corpus = os.path.join(workdir, 'corpus_prefixed.txt')
    lines = prefixed(train_txt, pref_corpus)
    print(f'prefixed {lines:,} training lines\n')

    results = {}
    for label, corpus in (('baseline (no prefix)', train_txt),
                          ('dummy prefix (SP-style)', pref_corpus)):
        print(f'--- training {label} ...', flush=True)
        model = train(corpus, merges, os.path.join(workdir, f'{label.split()[0]}.json'))
        bpt = score(model, heldout)
        results[label] = bpt
        print(f'{label:24} {bpt:.3f} bytes/token\n')

    base = results['baseline (no prefix)']
    pref = results['dummy prefix (SP-style)']
    delta = 100 * (base - pref) / base
    verdict = 'ADOPT' if delta > 0.5 else 'REJECT'
    print(f'{"baseline":24} {base:.3f}')
    print(f'{"dummy prefix":24} {pref:.3f}')
    print(f'{"improvement":24} {delta:+.2f}%')
    print(f'\nthreshold is >0.5% -> {verdict}')
    print('Adopting invalidates every model trained so far: chunk boundaries shift,')
    print('so it only applies to newly trained models.')
    with open(os.path.join(workdir, 'result.json'), 'w') as f:
        json.dump(results | {'delta_pct': delta, 'verdict': verdict}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())