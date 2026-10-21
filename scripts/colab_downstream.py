#!/usr/bin/env python3
"""Build the corpora on a Colab VM and run the tokenizer comparison there.

Everything is fetched from public sources on the VM, so the run needs no upload
and reproduces from the published artifacts:

  wiki         the corpus this repo publishes, from the Hugging Face Hub
  IndicCorp    a byte range of IndicCorpV2 from the Hub
  transcripts  the JSONL of Tamil video captions, from GitHub

Then it trains the same small model with each tokenizer, on the same corpus,
for the same number of tokens, and reports bits per byte on two held-out
splits.

Usage on a Colab VM:
  python colab_downstream.py --tokens 20000000 --gpu
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')

WIKI_URL = ('https://huggingface.co/barryallen16/tamil-tokenizer-wiki/resolve/main/'
            'tamil_wiki_corpus.txt')
TRANSCRIPTS_URL = ('https://github.com/barryallen16/sivabharani-comments/raw/'
                    'refs/heads/main/transcripts/output/tamil_transcripts.jsonl')


def run(cmd, **kw):
    print(f'$ {" ".join(cmd)}', flush=True)
    return subprocess.run(cmd, check=True, **kw)


def sh(cmd, **kw):
    return run(cmd, shell=True, **kw)


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'  have {dest}')
        return dest
    print(f'  downloading {url}')
    run(['curl', '-sL', '-o', dest, url])
    return dest


def build_corpora():
    """wiki, IndicCorp and transcripts, cleaned and split, then mixed."""
    os.makedirs(DATA, exist_ok=True)
    for d in ('wiki', 'indiccorp', 'transcripts', 'mixed'):
        os.makedirs(os.path.join(DATA, d), exist_ok=True)

    wiki_raw = fetch(WIKI_URL, os.path.join(DATA, 'tamil_wiki_corpus.txt'))
    wiki_clean = os.path.join(DATA, 'tamil_wiki_clean.txt')
    if not os.path.exists(wiki_clean):
        run([sys.executable, os.path.join(ROOT, 'scripts', 'extract_wiki_fast.py'),
             '--clean', wiki_raw, wiki_clean])
    if not os.path.exists(os.path.join(DATA, 'wiki', 'test.txt')):
        run([sys.executable, os.path.join(ROOT, 'scripts', 'prep_corpus.py'),
             '--split', wiki_clean, os.path.join(DATA, 'wiki')])

    ic_raw = os.path.join(DATA, 'indiccorp_ta.txt')
    if not os.path.exists(os.path.join(DATA, 'indiccorp', 'test.txt')):
        # A byte range of the file, not the whole thing. The full Tamil file is
        # far larger than the slice this project trains on.
        run([sys.executable, os.path.join(ROOT, 'scripts', 'fetch_indiccorp.py'),
             '--limit-mb', '400', '--out', ic_raw])
        run([sys.executable, os.path.join(ROOT, 'scripts', 'prep_corpus.py'),
             ic_raw, os.path.join(DATA, 'indiccorp')])

    if not os.path.exists(os.path.join(DATA, 'transcripts', 'test.txt')):
        run([sys.executable, os.path.join(ROOT, 'scripts', 'fetch_transcripts.py'),
             '--url', TRANSCRIPTS_URL, '--out-dir', os.path.join(DATA, 'transcripts')])

    mixed = os.path.join(DATA, 'mixed', 'train.txt')
    if not os.path.exists(mixed):
        run([sys.executable, os.path.join(ROOT, 'scripts', 'prep_corpus.py'),
             '--mix', os.path.join(DATA, 'mixed'), '200', 'wiki+ic+transcripts'])

    # One space on every line, so the trainer only ever sees the spaced form of
    # a line-initial word. The same preprocessing the published models use.
    prefixed = os.path.join(DATA, 'mixed', 'train_prefixed.txt')
    if not os.path.exists(prefixed):
        with open(mixed, 'rb') as f, open(prefixed, 'wb') as g:
            n = 0
            for line in f:
                g.write(b' ' + line if line.strip() else line)
                n += 1
        print(f'  prefixed {n} lines')
    return prefixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=20_000_000)
    ap.add_argument('--bytes', type=int, default=0,
                    help='match training bytes instead of training tokens')
    ap.add_argument('--train-mb', type=int, default=200)
    ap.add_argument('--eval-mb', type=int, default=3)
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--d-model', type=int, default=384)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--tokenizers', default='ours,indictok,mbert')
    ap.add_argument('--skip-data', action='store_true')
    ap.add_argument('--classify', action='store_true',
                    help='after each language-model run, fine-tune the saved '
                         'model on the Tamil news classification task')
    ap.add_argument('--cls-seeds', default='1234,1235,1236')
    ap.add_argument('--cls-epochs', type=int, default=4)
    ap.add_argument('--cls-batch', type=int, default=64)
    ap.add_argument('--cls-lr', type=float, default=1e-3)
    ap.add_argument('--cls-max-len', type=int, default=64)
    args = ap.parse_args()

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if not args.skip_data:
        print('=== building corpora ===', flush=True)
        build_corpora()

    train_file = os.path.join(DATA, 'mixed', 'train_prefixed.txt')
    evals = [os.path.join(DATA, 'wiki', 'test.txt'),
             os.path.join(DATA, 'transcripts', 'test.txt')]

    for name in args.tokenizers.split(','):
        print(f'\n=== {name} ===', flush=True)
        checkpoint = os.path.join(tempfile.gettempdir(), f'{name}_lm.pt')
        cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'downstream_lm.py'),
               '--tokenizer', name, '--tokens', str(args.tokens),
               '--train-file', train_file, '--train-mb', str(args.train_mb),
               '--eval-mb', str(args.eval_mb), '--context', str(args.context),
               '--d-model', str(args.d_model), '--layers', str(args.layers),
               '--heads', str(args.heads), '--batch', str(args.batch),
               '--lr', str(args.lr), '--seed', str(args.seed)]
        if args.classify:
            cmd += ['--save-checkpoint', checkpoint]
        if args.bytes:
            cmd += ['--bytes', str(args.bytes)]
        for e in evals:
            cmd += ['--eval-file', e]
        start = time.time()
        run(cmd)
        print(f'  {name} took {time.time() - start:.0f}s', flush=True)

        if args.classify:
            print(f'\n=== {name}: classification ===', flush=True)
            cls = [sys.executable, os.path.join(ROOT, 'scripts',
                                                'classify_tamil.py'),
                   '--tokenizer', name, '--checkpoint', checkpoint,
                   '--seeds', args.cls_seeds, '--epochs', str(args.cls_epochs),
                   '--batch', str(args.cls_batch), '--lr', str(args.cls_lr),
                   '--max-len', str(args.cls_max_len)]
            start = time.time()
            run(cls)
            print(f'  {name} classification took {time.time() - start:.0f}s',
                  flush=True)
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    print('\n=== comparison ===', flush=True)
    report = [sys.executable, os.path.join(ROOT, 'scripts', 'downstream_lm.py'), '--report']
    if args.bytes:
        report.append('--byte-run')
    run(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
