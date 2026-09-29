#!/usr/bin/env python3
"""BPE vs Unigram, measured on the same corpora.

Trains SentencePiece `unigram` and `bpe` at a fixed vocabulary on one training
corpus, scores both plus our own BPE model, and prints one table.

SentencePiece is the reference Unigram implementation. Using it instead of a
hand-rolled trainer matters: the algorithm is suffix-array seeding, forward-
backward marginals in the E-step, a digamma M-step and loss-based pruning, and
getting any of those wrong moves compression by tens of percent in either
direction. See `IMPROVEMENT_ROADMAP.md` for what that looked like in practice.

Usage:
  python scripts/spm_ab.py                       # 32K, wiki train, both splits
  python scripts/spm_ab.py --vocab-size 64000
  python scripts/spm_ab.py --train data/mixed/train.txt --label mixed
  python scripts/spm_ab.py --skip-train          # reuse existing .model files
"""
import argparse
import os
import subprocess
import sys
import time

import sentencepiece as spm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = [
    ('wiki held-out', 'data/wiki/test.txt'),
    ('IndicCorp held-out', 'data/indiccorp/test.txt'),
]


def train(corpus: str, out_prefix: str, model_type: str, vocab_size: int,
          seed_vocab_size: int) -> str:
    path = out_prefix + '.model'
    if os.path.exists(path):
        print(f'  reusing {path}')
        return path
    t = time.time()
    print(f'  training {model_type} {vocab_size} ...', flush=True)
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        # Lossless: without this, any character outside the corpus is UNK and
        # silently dropped, which would flatter whichever tokenizer saw more of
        # the corpus.
        byte_fallback=True,
        # Both trainers must start from the same overcomplete candidate set or
        # the comparison is confounded by the seed, not the objective.
        seed_sentencepiece_size=seed_vocab_size,
        # Tamil has no case, and NFKC would fold the ASCII and Tamil digits into
        # one another; identity keeps the corpus as written.
        normalization_rule_name='identity',
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=False,
        minloglevel=1,
    )
    print(f'    done in {time.time() - t:.1f}s -> {path}')
    return path


def score_spm(model_path: str, corpus: str, max_bytes: int):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    bytes_n = tokens = words = 0
    with open(corpus, encoding='utf-8') as f:
        for line in f:
            # Strip the line ending only. `score.rs` counts the line as written,
            # so stripping leading or trailing spaces here would give the two
            # columns different denominators and flatter whichever tokenizer
            # happened to be measured by this branch.
            line = line.rstrip('\r\n')
            if not line.strip():
                continue
            if bytes_n and bytes_n + len(line.encode('utf-8')) > max_bytes:
                break
            bytes_n += len(line.encode('utf-8'))
            tokens += len(sp.encode(line, out_type=int))
            words += len(line.split())
    return bytes_n / tokens, tokens / max(words, 1)


def score_ours(tokenizer_path: str, corpus: str, max_bytes: int):
    """Read bytes/token/tokens/word out of our `score` binary."""
    exe = os.path.join(BASE, 'target', 'release', 'score.exe')
    if not os.path.exists(exe):
        exe = os.path.join(BASE, 'target', 'release', 'score')
    if not os.path.exists(exe):
        return None
    out = subprocess.run(
        [exe, tokenizer_path, corpus, 'x', str(max_bytes)],
        capture_output=True, text=True, encoding='utf-8')
    if out.returncode != 0:
        return None
    # score prints: label \t MB \t tokens \t bytes/token \t tokens/word \t MB/s
    #
    # Read bytes/token straight off the line rather than rebuilding it from the
    # MB column. That column is printed to one decimal, so `int(float(p[1]) *
    # 1024 * 1024)` overshoots by up to 0.05 MB and inflates this row by ~0.4% --
    # in favour of our own model, which is the direction that matters.
    p = out.stdout.strip().split('\t')
    return float(p[3]), float(p[4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/wiki/train.txt')
    ap.add_argument('--label', default='wiki')
    ap.add_argument('--vocab-size', type=int, default=32000)
    ap.add_argument('--seed-vocab-size', type=int, default=250000)
    ap.add_argument('--max-bytes', type=int, default=6 * 1024 * 1024)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--ours', default='output/v3_wiki_30000.json')
    args = ap.parse_args()

    corpus = args.train if os.path.isabs(args.train) else os.path.join(BASE, args.train)
    outdir = os.path.join(BASE, 'output', f'spm_{args.label}_{args.vocab_size}')
    os.makedirs(outdir, exist_ok=True)

    models = {}
    for model_type in ('unigram', 'bpe'):
        if args.skip_train:
            path = os.path.join(outdir, f'{model_type}.model')
            if not os.path.exists(path):
                print(f'missing {path}; drop --skip-train')
                return 1
            models[model_type] = path
            continue
        models[model_type] = train(
            corpus, os.path.join(outdir, model_type), model_type,
            args.vocab_size, args.seed_vocab_size)

    print()
    header = f'{"model":<22}' + ''.join(f'{name:>26}' for name, _ in SPLITS)
    print(f'corpus: {args.train}  vocab: {args.vocab_size}')
    print(header)
    print('-' * len(header))

    rows = []
    for model_type in ('unigram', 'bpe'):
        rows.append((f'sentencepiece {model_type}', models[model_type], 'spm'))
    ours = os.path.join(BASE, args.ours) if not os.path.isabs(args.ours) else args.ours
    if os.path.exists(ours):
        rows.append(('ours (bpe)', ours, 'ours'))
    else:
        print(f'note: {args.ours} not found, skipping our model')

    for label, path, kind in rows:
        cells = []
        for split_name, rel in SPLITS:
            split = os.path.join(BASE, rel)
            if not os.path.exists(split):
                cells.append(f'{"n/a":>26}')
                continue
            res = score_spm(path, split, args.max_bytes) if kind == 'spm' \
                else score_ours(path, split, args.max_bytes)
            if res is None:
                cells.append(f'{"n/a":>26}')
                continue
            bytes_per_token, tok_per_word = res
            cells.append(f'{bytes_per_token:>18.3f} {tok_per_word:>7.2f}')
        print(f'{label:<22}' + ''.join(cells))

    print()
    print('bytes/token (higher is better) and tokens/word (lower is better)')
    return 0


if __name__ == '__main__':
    sys.exit(main())