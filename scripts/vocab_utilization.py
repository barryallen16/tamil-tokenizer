#!/usr/bin/env python3
"""How much of a trained vocabulary actually earns its keep.

The roadmap recommends 120K merges but never measured whether the tail is used.
A tokenizer whose tail is inert could be smaller at equal compression and
higher throughput, which would change the recommended default.

Usage:
  python scripts/vocab_utilization.py <model.json> [corpus ...]

Corpora default to the two held-out splits. Each model must be listed before the
corpora it should be scored on.
"""
import json
import subprocess
import sys
from collections import Counter


def score(model_path, corpora):
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    vocab = model['vocab']
    inv = {i: t for t, i in vocab.items()}

    lines = []
    for path in corpora:
        with open(path, encoding='utf-8') as f:
            lines += [l.rstrip('\n') for l in f if l.strip()]

    proc = subprocess.run(
        ['./target/release/encode_lines.exe', model_path],
        input='\n'.join(lines),
        capture_output=True, text=True, encoding='utf-8',
    )
    if proc.returncode != 0:
        sys.exit('encode_lines failed: ' + proc.stderr[:400])

    counts = Counter()
    for ids in json.loads(proc.stdout).values():
        counts.update(ids)

    return vocab, inv, counts, len(lines), model['merges']


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    corpora = sys.argv[2:] or ['data/wiki/test.txt', 'data/indiccorp/test.txt']
    model_path = sys.argv[1]

    vocab, inv, counts, n_lines, merges = score(model_path, corpora)
    total = len(vocab)
    learned = {i for i in inv if i not in counts}

    used = {i: c for i, c in counts.items() if i in inv}
    unused = total - len(used)
    once = sum(1 for c in used.values() if c == 1)
    rare = sum(1 for c in used.values() if c <= 10)

    print(f'model            {model_path}')
    print(f'corpora          {", ".join(corpora)}')
    print(f'lines scored     {n_lines:,}')
    print(f'vocab size       {total:,}')
    print()
    print(f'never used       {unused:>8,}  ({100 * unused / total:5.1f}%)')
    print(f'used exactly 1x  {once:>8,}  ({100 * once / total:5.1f}%)')
    print(f'used <= 10x      {rare:>8,}  ({100 * rare / total:5.1f}%)')
    print(f'used > 10x       {total - rare:>8,}  ({100 * (total - rare) / total:5.1f}%)')

    # Which merge ranks carry the weight. Merges are ordered, so rank position
    # is the honest axis: a tail that is all high-rank merges is all low-value.
    buckets = [(0, 30000), (30000, 60000), (60000, 90000), (90000, len(merges))]
    print('\nmerge-rank distribution of used tokens')
    for lo, hi in buckets:
        ids = {vocab[merges[r][0] + merges[r][1]] for r in range(lo, min(hi, len(merges)))
               if merges[r][0] + merges[r][1] in vocab}
        if not ids:
            continue
        hit = sum(1 for i in ids if i in counts)
        share = sum(counts[i] for i in ids if i in counts)
        pct = 100 * share / max(sum(counts.values()), 1)
        print(f'  merges {lo:>6,}-{hi:>6,}  {len(ids):>7,} tokens  '
              f'{hit:>7,} used ({100 * hit / len(ids):5.1f}%)  '
              f'{pct:5.1f}% of all emitted tokens')

    print('\nleast-used tokens (10)')
    for i, _ in sorted(used.items(), key=lambda kv: kv[1])[:10]:
        print(f'  {counts[i]:>10,}  {inv[i]!r}')

    print('\nunused, non-reserved (20 of %d)' % len(learned))
    shown = 0
    for i in sorted(vocab.values(), key=lambda x: inv[x]):
        t = inv[i]
        if i in used or t.startswith('<|') or not t:
            continue
        print(f'  {t!r}')
        shown += 1
        if shown >= 20:
            break

    tail_cut = _find_cutoff(used, inv, total)
    print(f'\nvocab actually exercised (>10 uses): {total - rare:,}')
    print(f'smallest vocab keeping 99.9% of emitted tokens: {tail_cut:,}')
    return 0


def _find_cutoff(used, inv, total):
    """Smallest vocab size that still covers 99.9% of emitted tokens.

    Sorts tokens by frequency and keeps the head. Approximate because the
    tokenizer emits by rank order, not frequency, but it bounds the question.
    """
    ranked = sorted(used.items(), key=lambda kv: -kv[1])
    target = sum(c for _, c in ranked) * 0.999
    seen = 0
    for n, (_, c) in enumerate(ranked, 1):
        seen += c
        if seen >= target:
            return n
    return len(ranked)


if __name__ == '__main__':
    sys.exit(main())