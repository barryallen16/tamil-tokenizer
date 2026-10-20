#!/usr/bin/env python3
"""Compare the classification runs, with a paired bootstrap on the test set.

Reads the `{tokenizer}_cls.json` files that classify_tamil.py writes and
reports accuracy and macro F1 per tokenizer plus the paired difference against
ours. The bootstrap resamples test headlines, so the interval is about this
test set, not about new headlines.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, 'results', 'downstream')
NAMES = ['ours', 'indictok', 'mbert']


def macro_f1(y, pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(scores))


def load(directory, names):
    runs = {}
    for name in names:
        path = os.path.join(directory, f'{name}_cls.json')
        if not os.path.exists(path):
            print(f'missing {path}', file=sys.stderr)
            return None
        with open(path, encoding='utf-8') as f:
            runs[name] = json.load(f)
    labels = np.array(runs[names[0]]['test_labels'])
    for name in names[1:]:
        assert np.array_equal(labels, np.array(runs[name]['test_labels'])), name
    return runs


def paired_bootstrap(diff, reps, seed):
    rng = np.random.default_rng(seed)
    n = len(diff)
    chunks = []
    done = 0
    while done < reps:
        size = min(1000, reps - done)
        idx = rng.integers(0, n, size=(size, n))
        chunks.append(diff[idx].mean(1))
        done += size
    means = np.concatenate(chunks)
    return float(diff.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR)
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    runs = load(args.dir, NAMES)
    if runs is None:
        return 1
    y = np.array(runs['ours']['test_labels'])
    n_classes = len(runs['ours']['labels'])

    print(f'{"tokenizer":10s} {"tokens/headline":>15s} {"accuracy":>19s} '
          f'{"macro F1":>19s}')
    for name in NAMES:
        run = runs[name]
        preds = [np.array(s['test_pred']) for s in run['seeds']]
        accs = [float((p == y).mean()) for p in preds]
        f1s = [macro_f1(y, p, n_classes) for p in preds]
        print(f'{name:10s} {run["tokens_per_headline"]["test_mean"]:>15.2f} '
              f'{np.mean(accs):>8.4f} ({min(accs):.4f}-{max(accs):.4f}) '
              f'{np.mean(f1s):>8.4f} ({min(f1s):.4f}-{max(f1s):.4f})')
    print(f'{"majority":10s} {"":>15s} {runs["ours"]["majority_accuracy"]:>8.4f}')

    print()
    for other in ('indictok', 'mbert'):
        ours = [np.array(s['test_pred']) for s in runs['ours']['seeds']]
        theirs = [np.array(s['test_pred']) for s in runs[other]['seeds']]
        pairs = [(a == y).astype(float) - (b == y).astype(float)
                 for a in ours for b in theirs]
        diff = np.mean(pairs, axis=0)
        mean, low, high = paired_bootstrap(diff, args.reps, args.seed)
        print(f'ours - {other:9s} {mean * 100:+.2f} points  '
              f'95% CI [{low * 100:+.2f}, {high * 100:+.2f}]  '
              f'{"outside zero" if low > 0 or high < 0 else "includes zero"}')

    print()
    print('per-class F1, mean over seeds')
    for name in NAMES:
        f1s = np.mean([s['test_f1_per_class'] for s in runs[name]['seeds']],
                      axis=0)
        print(f'{name:10s} ' +
              '  '.join(f'{label} {f1:.3f}' for label, f1
                        in zip(runs[name]['labels'], f1s)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
