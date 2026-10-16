#!/usr/bin/env python3
"""Train the same small model with different tokenizers and compare.

The compression claim says that a tokenizer needing fewer pieces for the same
text lets a model cover more text inside a fixed training budget, and so ends
up better. This measures it: one model, one corpus, one training budget, three
tokenizers.

The comparison is reported in bits per byte. Loss per token cannot be compared
across tokenizers, because a tokenizer that packs more text into a token has
fewer tokens to spread the loss over. Bits per byte is computed over the same
bytes for every run, so it means the same thing each time.

Two budgets are available:

  --tokens N   the same number of training tokens for every tokenizer, so the
               tokenizer that packs more text per token sees more of the corpus.
               This is the compression hypothesis, and it is the one a real
               training run faces.
  --bytes N    the same number of training bytes for every tokenizer, so every
               tokenizer sees the same text. This isolates the quality of the
               segmentation from its compression.

Usage:
  python scripts/downstream_lm.py --tokenizer ours --tokens 20000000
  python scripts/downstream_lm.py --tokenizer mbert --tokens 20000000
  python scripts/downstream_lm.py --tokenizer indictok --tokens 20000000
  python scripts/downstream_lm.py --report

Requires `pip install torch transformers tokenizers`.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'output', 'downstream')

# The three tokenizers. Ours is the published model; the other two are the
# strongest general-purpose Tamil tokenizers available on the Hub.
TOKENIZERS = {
    'ours': ('barryallen16/tamil-tokenizer-wiki', 'tamil_tokenizer_wiki_120k_mixed'),
    'indictok': ('vivekvar/indictok-256k', None),
    'mbert': ('bert-base-multilingual-cased', None),
}


def load_tokenizer(name):
    """Return a tokenizer that produces ids only, with no added tokens."""
    from transformers import AutoTokenizer

    repo, subfolder = TOKENIZERS[name]
    return AutoTokenizer.from_pretrained(repo, subfolder=subfolder)


def tokenize_lines(tok, lines, label):
    """Tokenize once and cache the ids, since this is the slow part."""
    cache = os.path.join(RESULTS, f'{label}.npy')
    if os.path.exists(cache):
        return np.load(cache)
    os.makedirs(RESULTS, exist_ok=True)
    ids = []
    for i, line in enumerate(lines):
        ids.extend(tok(line, add_special_tokens=False)['input_ids'])
        if (i + 1) % 20000 == 0:
            print(f'  {label}: {i + 1}/{len(lines)} lines', flush=True)
    arr = np.array(ids, dtype=np.int32)
    np.save(cache, arr)
    return arr


def build_model(vocab_size, d_model, n_layers, n_heads, context):
    """A small pre-LN causal transformer. Same shape for every tokenizer."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = n_heads
            self.dh = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            b, t, c = x.shape
            q, k, v = self.qkv(x).split(c, dim=2)
            q = q.view(b, t, self.h, self.dh).transpose(1, 2)
            k = k.view(b, t, self.h, self.dh).transpose(1, 2)
            v = v.view(b, t, self.h, self.dh).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).reshape(b, t, c))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = Attention()
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class MiniLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(context, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embed.weight  # tied
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        def forward(self, idx):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.embed(idx) + self.pos(pos)[None]
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

    return MiniLM()


def train(model, tokens, budget, context, batch, lr, seed, log_every=50):
    import torch

    torch.manual_seed(seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    steps = max(1, budget // (batch * (context - 1)))
    warmup = max(1, steps // 20)

    def lr_at(step):
        if step < warmup:
            return lr * step / warmup
        p = (step - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    order = np.random.permutation(len(tokens))
    cursor = 0
    model.train()
    start = time.time()
    running = 0.0
    for step in range(steps):
        for g in optimiser.param_groups:
            g['lr'] = lr_at(step)
        batch_idx = []
        for _ in range(batch):
            if cursor + context > len(order):
                order = np.random.permutation(len(tokens))
                cursor = 0
            batch_idx.append(order[cursor:cursor + context])
            cursor += context
        x = torch.from_numpy(np.stack(batch_idx).astype(np.int64))
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        running += loss.item()
        if (step + 1) % log_every == 0:
            done = (step + 1) * batch * (context - 1)
            print(f'  step {step + 1}/{steps} loss {running / log_every:.4f} '
                  f'tokens {done / 1e6:.2f}M elapsed {time.time() - start:.0f}s', flush=True)
            running = 0.0
    return time.time() - start


def evaluate(model, tokens, context, batch, limit):
    """Total negative log likelihood in nats, over the tokens given."""
    import torch

    model.eval()
    nll = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, limit - context, batch * context):
            chunk = tokens[i:i + batch * context]
            rows = chunk[:min(len(chunk) // context, batch) * context]
            if len(rows) < context:
                break
            x = torch.from_numpy(rows.astype(np.int64)).view(-1, context)
            logits = model(x[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                reduction='sum')
            nll += loss.item()
            count += x[:, 1:].numel()
    return nll, count


def read_lines(path, limit_bytes=None):
    lines = []
    size = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            lines.append(line)
            size += len(line.encode('utf-8')) + 1
            if limit_bytes and size >= limit_bytes:
                break
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', choices=list(TOKENIZERS), default='ours')
    ap.add_argument('--tokens', type=int, default=1_000_000)
    ap.add_argument('--bytes', type=int, default=0, help='training bytes, overrides --tokens')
    ap.add_argument('--train-file', default=os.path.join(ROOT, 'data', 'mixed', 'train.txt'))
    ap.add_argument('--eval-file', action='append', default=None,
                    help='held-out file to score, repeatable')
    ap.add_argument('--train-mb', type=int, default=24)
    ap.add_argument('--eval-mb', type=int, default=3)
    ap.add_argument('--context', type=int, default=256)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    if not args.eval_file:
        args.eval_file = [os.path.join(ROOT, 'data', 'wiki', 'test.txt')]

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.makedirs(RESULTS, exist_ok=True)

    if args.report:
        report()
        return 0

    import torch

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    budget_kind = 'bytes' if args.bytes else 'tokens'
    print(f'tokenizer={args.tokenizer} budget={budget_kind} {args.bytes or args.tokens}')
    tok = load_tokenizer(args.tokenizer)

    train_lines = read_lines(args.train_file, args.train_mb * 1024 * 1024)
    evals = []
    for path in args.eval_file:
        lines = read_lines(path, args.eval_mb * 1024 * 1024)
        label = os.path.splitext(os.path.basename(path))[0]
        evals.append((label, lines, path))
    print(f'  train {len(train_lines)} lines, eval ' +
          ', '.join(f'{len(l)} lines' for _, l, _ in evals))

    train_ids = tokenize_lines(tok, train_lines, f'train_{args.tokenizer}')
    eval_ids = []
    for label, lines, path in evals:
        ids = tokenize_lines(tok, lines, f'eval_{label}_{args.tokenizer}')
        nbytes = sum(len(l.encode('utf-8')) + 1 for l in lines)
        eval_ids.append((label, ids, nbytes))
        print(f'  eval {label}: {len(ids)} tokens over {nbytes} bytes')

    vocab = tok.vocab_size
    model = build_model(vocab, args.d_model, args.layers, args.heads, args.context)
    params = sum(p.numel() for p in model.parameters())
    print(f'  vocab {vocab}, parameters {params / 1e6:.1f}M')

    # Bytes per token, measured on the held-out text. It turns a byte budget
    # into the number of training tokens that cover it.
    bytes_per_token = (sum(b for _, _, b in eval_ids) /
                       max(1, sum(len(i) for _, i, _ in eval_ids)))
    if args.bytes:
        # Same text for every tokenizer: take the tokens that cover the budget.
        budget_tokens = min(int(args.bytes / bytes_per_token), len(train_ids))
    else:
        budget_tokens = min(args.tokens, len(train_ids))

    elapsed = train(model, train_ids, budget_tokens, args.context, args.batch, args.lr, args.seed)

    scores = {}
    for label, ids, nbytes in eval_ids:
        nll, count = evaluate(model, ids, args.context, args.batch, len(ids))
        scores[label] = {
            'bits_per_byte': nll / math.log(2) / nbytes,
            'nll_nats_per_token': nll / max(1, count),
            'tokens_per_byte': len(ids) / nbytes,
            'eval_tokens': count,
            'eval_bytes': nbytes,
        }

    result = {
        'tokenizer': args.tokenizer,
        'vocab': vocab,
        'parameters': params,
        'train_tokens': budget_tokens,
        'train_bytes_seen': round(budget_tokens * bytes_per_token),
        'scores': scores,
        'seconds': elapsed,
        'args': vars(args),
    }
    path = os.path.join(RESULTS, f'{args.tokenizer}.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  {args.tokenizer}: {budget_tokens} training tokens, {elapsed:.0f}s')
    for label, s in scores.items():
        print(f'    {label}: bits/byte {s["bits_per_byte"]:.4f}  '
              f'nats/token {s["nll_nats_per_token"]:.4f}  '
              f'tokens/byte {s["tokens_per_byte"]:.4f}')
    print(f'  wrote {path}')
    return 0


def report():
    """Print the comparison."""
    rows = []
    for name in TOKENIZERS:
        path = os.path.join(RESULTS, f'{name}.json')
        if os.path.exists(path):
            rows.append(json.load(open(path)))
    if not rows:
        print('no results yet')
        return
    labels = list(rows[0].get('scores', {}).keys())
    head = f'{"tokenizer":12} {"vocab":>8} {"params":>9} {"train tok":>10} {"sec":>6}'
    for label in labels:
        head += f' | {label} bpb  tok/byte'
    print(head)
    for r in sorted(rows, key=lambda r: r['scores'][labels[0]]['bits_per_byte']):
        line = (f'{r["tokenizer"]:12} {r["vocab"]:>8} {r["parameters"] / 1e6:>8.1f}M '
                f'{r["train_tokens"]:>10} {r["seconds"]:>6.0f}')
        for label in labels:
            s = r['scores'][label]
            line += f' | {s["bits_per_byte"]:6.4f}  {s["tokens_per_byte"]:.4f}'
        print(line)
    for label in labels:
        best = min(rows, key=lambda r: r['scores'][label]['bits_per_byte'])
        others = [r for r in rows if r is not best]
        if not others:
            continue
        print(f'\n{label}: {best["tokenizer"]} is best at '
              f'{best["scores"][label]["bits_per_byte"]:.4f} bits/byte')
        for r in others:
            b = best['scores'][label]['bits_per_byte']
            o = r['scores'][label]['bits_per_byte']
            print(f'  {100 * (o - b) / b:+.1f}% vs {r["tokenizer"]}')


if __name__ == '__main__':
    sys.exit(main())
