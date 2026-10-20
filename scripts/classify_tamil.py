#!/usr/bin/env python3
"""Fine-tune the pretrained small model on Tamil news headlines.

The data, the model and the training schedule are the same for every
tokenizer; only the tokenizer changes. The task is the MTEB
TamilNewsClassification set: 14,521 Tamil headlines in six classes, with an
official test split of 2,048.

The encoder is the one downstream_lm.py trained for the bits-per-byte
comparison, so this asks whether the tokenizer that compresses best also
transfers best to a real classification task.

Usage:
  python classify_tamil.py --tokenizer ours --checkpoint /tmp/ours.pt
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

from downstream_lm import RESULTS, build_model, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'classification')
BASE = ('https://huggingface.co/datasets/mteb/TamilNewsClassification/'
        'resolve/main/data/')
FILES = {'train': 'train-00000-of-00001.parquet',
         'test': 'test-00000-of-00001.parquet'}
# The labels are alphabetical in the MTEB export; the source is the headline
# scraper behind github.com/vanangamudi/tamil-news-classification.
LABELS = ['cinema', 'india', 'politics', 'sports', 'tamilnadu', 'world']


def fetch(split):
    """Return the texts and labels for a split, caching the parquet."""
    import pandas as pd

    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f'{split}.parquet')
    if not os.path.exists(path):
        import urllib.request
        print(f'  downloading {BASE + FILES[split]}', flush=True)
        urllib.request.urlretrieve(BASE + FILES[split], path)
    df = pd.read_parquet(path)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].to_numpy(dtype=np.int64)
    return texts, labels


def encode(tok, texts, max_len):
    ids = []
    for text in texts:
        ids.append(tok(text, add_special_tokens=False)['input_ids'][:max_len])
    return ids


def stratified_split(labels, frac, seed):
    """Indices for a validation split, drawn per class."""
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        val.extend(idx[:max(1, int(round(len(idx) * frac)))].tolist())
    val = np.array(sorted(val))
    mask = np.ones(len(labels), dtype=bool)
    mask[val] = False
    return np.flatnonzero(mask), val


def batches(ids, labels, batch, shuffle, rng):
    import torch

    n = len(ids)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch):
        idx = order[start:start + batch]
        lengths = np.array([len(ids[i]) for i in idx], dtype=np.int64)
        width = max(1, int(lengths.max()))
        x = np.zeros((len(idx), width), dtype=np.int64)
        for row, i in enumerate(idx):
            x[row, :len(ids[i])] = ids[i]
        yield (torch.from_numpy(x), torch.from_numpy(labels[idx]),
               torch.from_numpy(lengths))


def make_classifier(encoder, d_model, n_classes, dropout=0.1):
    import torch
    import torch.nn as nn

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(d_model, n_classes)

        def forward(self, x, lengths):
            h = self.encoder.hidden(x)
            positions = torch.arange(x.shape[1], device=x.device)[None, :]
            mask = (positions < lengths[:, None]).float().unsqueeze(-1)
            pooled = (h * mask).sum(1) / lengths[:, None]
            return self.out(self.drop(pooled))

    return Classifier()


def predict(model, ids, labels, batch, device):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for x, _, lengths in batches(ids, labels, batch, False, None):
            logits = model(x.to(device), lengths.to(device))
            out.extend(logits.argmax(-1).cpu().tolist())
    return np.array(out)


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
    return float(np.mean(scores)), [round(s, 4) for s in scores]


def fit_seed(state, vocab, arch, train, val, test, args, seed, device):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    encoder = build_model(vocab, arch['d_model'], arch['layers'],
                          arch['heads'], arch['context'])
    if state is not None:
        encoder.load_state_dict(state)
    model = make_classifier(encoder, arch['d_model'], len(LABELS)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.98))
    steps = math.ceil(len(train[0]) / args.batch) * args.epochs
    warmup = max(1, steps // 10)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    step = 0
    best = {'f1': -1.0, 'epoch': -1, 'state': None, 'val_acc': 0.0}
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y, lengths in batches(train[0], train[1], args.batch, True, rng):
            for group in optimiser.param_groups:
                group['lr'] = lr_at(step)
            step += 1
            optimiser.zero_grad(set_to_none=True)
            logits = model(x.to(device), lengths.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(loss.item())
        val_pred = predict(model, val[0], val[1], args.batch, device)
        f1, _ = macro_f1(val[1], val_pred, len(LABELS))
        val_acc = float((val_pred == val[1]).mean())
        print(f'    epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.4f}  '
              f'val acc {val_acc:.4f}  val macro F1 {f1:.4f}', flush=True)
        if f1 > best['f1']:
            best = {'f1': f1, 'epoch': epoch + 1, 'val_acc': val_acc,
                    'state': {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
    model.load_state_dict(best['state'])
    test_pred = predict(model, test[0], test[1], args.batch, device)
    test_acc = float((test_pred == test[1]).mean())
    test_f1, per_class = macro_f1(test[1], test_pred, len(LABELS))
    return {
        'seed': seed,
        'best_epoch': best['epoch'],
        'val_accuracy': round(best['val_acc'], 4),
        'val_macro_f1': round(best['f1'], 4),
        'test_accuracy': round(test_acc, 4),
        'test_macro_f1': round(test_f1, 4),
        'test_f1_per_class': per_class,
        'test_pred': test_pred.tolist(),
        'seconds': round(time.time() - start, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', choices=['ours', 'indictok', 'mbert'],
                    default='ours')
    ap.add_argument('--checkpoint', default='',
                    help='downstream_lm.py checkpoint; "none" trains from scratch')
    ap.add_argument('--out', default='')
    ap.add_argument('--seeds', default='1234,1235,1236')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--max-len', type=int, default=64)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke test: keep this many train and test rows')
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--d-model', type=int, default=384)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    args = ap.parse_args()

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = load_tokenizer(args.tokenizer)
    vocab = len(tok)

    state = None
    arch = {'d_model': args.d_model, 'layers': args.layers,
            'heads': args.heads, 'context': args.context}
    pretrain = {}
    if args.checkpoint and args.checkpoint != 'none':
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt['state_dict']
        vocab = ckpt['vocab']
        saved = ckpt.get('args', {})
        arch = {'d_model': saved.get('d_model', args.d_model),
                'layers': saved.get('layers', args.layers),
                'heads': saved.get('heads', args.heads),
                'context': saved.get('context', args.context)}
        pretrain = {'tokenizer': saved.get('tokenizer'),
                    'tokens': saved.get('tokens'), 'bytes': saved.get('bytes'),
                    'seed': saved.get('seed')}
        print(f'  checkpoint {args.checkpoint}: vocab {vocab}, arch {arch}, '
              f'pretrain {pretrain}', flush=True)

    texts, labels = fetch('train')
    test_texts, test_labels = fetch('test')
    if args.limit:
        texts, labels = texts[:args.limit], labels[:args.limit]
        test_texts, test_labels = test_texts[:args.limit], test_labels[:args.limit]
    train_idx, val_idx = stratified_split(labels, args.val_frac, 1234)
    train_ids = encode(tok, [texts[i] for i in train_idx], args.max_len)
    val_ids = encode(tok, [texts[i] for i in val_idx], args.max_len)
    test_ids = encode(tok, test_texts, args.max_len)
    train = (train_ids, labels[train_idx])
    val = (val_ids, labels[val_idx])
    test = (test_ids, test_labels)
    lengths_train = [len(x) for x in train_ids]
    lengths_test = [len(x) for x in test_ids]
    print(f'  train {len(train_ids)}  val {len(val_ids)}  test {len(test_ids)}  '
          f'tokens/headline {np.mean(lengths_train):.1f} train, '
          f'{np.mean(lengths_test):.1f} test, max {max(lengths_test)}', flush=True)

    majority = int(np.bincount(labels[train_idx], minlength=len(LABELS)).argmax())
    majority_acc = float((test_labels == majority).mean())

    runs = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        print(f'  fine-tune seed {seed}', flush=True)
        runs.append(fit_seed(state, vocab, arch, train, val, test, args, seed,
                             device))

    accs = [r['test_accuracy'] for r in runs]
    f1s = [r['test_macro_f1'] for r in runs]
    result = {
        'task': 'mteb/TamilNewsClassification',
        'labels': LABELS,
        'tokenizer': args.tokenizer,
        'checkpoint': args.checkpoint,
        'pretrain': pretrain,
        'arch': arch,
        'vocab': vocab,
        'data': {'train': len(train_ids), 'val': len(val_ids),
                 'test': len(test_ids), 'limit': args.limit},
        'tokens_per_headline': {
            'train_mean': round(float(np.mean(lengths_train)), 2),
            'test_mean': round(float(np.mean(lengths_test)), 2),
            'test_max': int(max(lengths_test)),
        },
        'majority_label': majority,
        'majority_accuracy': round(majority_acc, 4),
        'seeds': runs,
        'summary': {
            'test_accuracy_mean': round(float(np.mean(accs)), 4),
            'test_accuracy_min': round(float(np.min(accs)), 4),
            'test_accuracy_max': round(float(np.max(accs)), 4),
            'test_macro_f1_mean': round(float(np.mean(f1s)), 4),
            'test_macro_f1_min': round(float(np.min(f1s)), 4),
            'test_macro_f1_max': round(float(np.max(f1s)), 4),
        },
        'test_labels': test_labels.tolist(),
        'args': vars(args),
    }
    out = args.out or os.path.join(RESULTS, f'{args.tokenizer}_cls.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'\n  {args.tokenizer}: test accuracy '
          f'{result["summary"]["test_accuracy_mean"]:.4f} '
          f'({result["summary"]["test_accuracy_min"]:.4f}-'
          f'{result["summary"]["test_accuracy_max"]:.4f}), macro F1 '
          f'{result["summary"]["test_macro_f1_mean"]:.4f}, '
          f'majority {majority_acc:.4f}')
    print(f'  wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
