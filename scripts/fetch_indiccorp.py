#!/usr/bin/env python3
"""Fetch a slice of IndicCorpV2 Tamil into a plain text file.

The full `data/ta.txt` is 11.8 GB, so this pulls a leading byte range instead.
Range requests make that cheap, and the cap is enough to build a training
corpus: a tokenizer saturates long before 400 MB.

Usage:
  python scripts/fetch_indiccorp.py                 # 400 MB into data/indiccorp_ta.txt
  python scripts/fetch_indiccorp.py --limit-mb 800
  python scripts/fetch_indiccorp.py --out data/x.txt --limit-mb 200

Resumable: rerunning continues from whatever is already on disk.
"""
import argparse
import os
import sys
import time

import requests

URL = 'https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/ta.txt'
CHUNK = 8 * 1024 * 1024


def remote_size(session):
    r = session.head(URL, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers['Content-Length'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-mb', type=int, default=400)
    ap.add_argument('--out', default='data/indiccorp_ta.txt')
    args = ap.parse_args()

    want = args.limit_mb * 1024 * 1024
    session = requests.Session()

    total = remote_size(session)
    have = os.path.getsize(args.out) if os.path.exists(args.out) else 0
    if have >= min(want, total):
        print(f'{args.out}: already {have / 1e6:.0f} MB, nothing to do')
        return 0
    if have:
        print(f'{args.out}: resuming at {have / 1e6:.0f} MB')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    target = min(want, total)
    start = time.time()
    fetched = have

    with open(args.out, 'ab') as out:
        pos = have
        while pos < target:
            end = min(pos + CHUNK, target) - 1
            r = session.get(
                URL,
                headers={'Range': f'bytes={pos}-{end}'},
                stream=True,
                timeout=120,
            )
            if r.status_code not in (200, 206):
                print(f'HTTP {r.status_code} at byte {pos}', file=sys.stderr)
                return 1
            for part in r.iter_content(1024 * 1024):
                out.write(part)
                pos += len(part)
                fetched += len(part)
            r.close()
            done = min(pos, target) / min(want, total)
            rate = (fetched - have) / max(time.time() - start, 1e-9) / 1e6
            print(f'\r{pos / 1e6:7.0f} / {target / 1e6:.0f} MB  '
                  f'({done * 100:.0f}%)  {rate:.1f} MB/s', end='', flush=True)

    print(f'\nwrote {args.out}: {os.path.getsize(args.out) / 1e6:.0f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())