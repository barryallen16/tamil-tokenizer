#!/usr/bin/env python3
"""Fetch and clean Tamil video transcripts, split by video into train and held-out.

Why this corpus exists: every number this project has published comes from
Wikipedia or IndicCorp, which is formal written prose. Both the training
corpora and the held-out splits are prose, so a tokenizer tuned on them looks
better on them than it has any right to. Transcripts are what Tamil speakers
actually type and say, and they contain a form nothing else has: English words
with Tamil glued inside them, mid-word, such as `thatடiled` or `motorரolaலயோ`.

The split is by video, not by line. Captions from one video are contiguous
utterances of one speaker, so a line-level split would put near-duplicate
sentences on both sides and make the held-out score meaningless.

Usage:
  python scripts/fetch_transcripts.py                       # download and prepare
  python scripts/fetch_transcripts.py --local <path.jsonl>  # use a local copy
  python scripts/fetch_transcripts.py --test-share 0.2

Requires `requests` only for the download; the local path needs the standard
library alone.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = ('https://github.com/barryallen16/sivabharani-comments/raw/'
               'refs/heads/main/transcripts/output/tamil_transcripts.jsonl')

TAMIL = re.compile(r'[\u0B80-\u0BFF]')
LATIN = re.compile(r'[A-Za-z]')
ASCII_PUNCT = set('|_=<>{}[]#*/' + chr(92))

# The same gate `extract_wiki_fast.py` uses, with the caption artefacts added.
# `>>` is a speaker label, `[...]` is a caption annotation such as [இசை] for
# music or [சிரிப்பு] for laughter, and neither is text.
CLEANERS = [
    (re.compile(r'^\s*>>\s*'), ''),
    (re.compile(r'\[[^\]]*\]'), ' '),
    (re.compile(r'https?://\S+'), ' '),
    (re.compile(r'&[a-zA-Z]+;|&#\d+;'), ' '),
    (re.compile(r'\s+'), ' '),
]


def clean_caption(text):
    for pattern, repl in CLEANERS:
        text = pattern.sub(repl, text)
    return text.strip()


def is_tamil_prose(text):
    """True when the line reads as Tamil text rather than caption noise."""
    if len(text) < 30:
        return False
    letters = sum(1 for c in text if TAMIL.match(c))
    punct = sum(1 for c in text if c in ASCII_PUNCT)
    return letters >= len(text) * 0.4 and punct <= len(text) * 0.02


def download(url, dest):
    print(f'Downloading {url}')
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, 'wb') as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print(f'  {total / 1e6:.1f} MB -> {dest}')
    return dest


def captions_by_video(path):
    """Yield (video_id, [caption lines]) for every video in the file."""
    with io.open(path, encoding='utf-8') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            for video_id, items in data.items():
                if not isinstance(items, list):
                    continue
                lines = []
                for item in items:
                    if isinstance(item, dict):
                        text = item.get('text', '')
                    elif isinstance(item, str):
                        text = item
                    else:
                        continue
                    if isinstance(text, str) and text.strip():
                        lines.append(text)
                if lines:
                    yield video_id, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default=DEFAULT_URL)
    ap.add_argument('--local', help='use this JSONL instead of downloading')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'data', 'transcripts'))
    ap.add_argument('--test-share', type=float, default=0.2)
    ap.add_argument('--keep-raw', action='store_true',
                    help='keep the downloaded JSONL next to the split')
    args = ap.parse_args()

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = args.local or os.path.join(args.out_dir, 'raw.jsonl')
    if not args.local:
        download(args.url, raw_path)

    # Split by video. The hash keeps the split stable across runs and across
    # machines, without depending on the order of the file.
    train_path = os.path.join(args.out_dir, 'train.txt')
    test_path = os.path.join(args.out_dir, 'test.txt')
    train_lines, test_lines = [], []
    train_seen, test_seen = set(), set()
    videos = kept = dropped = 0

    for video_id, lines in captions_by_video(raw_path):
        videos += 1
        bucket = int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        is_test = bucket < args.test_share
        for line in lines:
            cleaned = clean_caption(line)
            if not is_tamil_prose(cleaned):
                dropped += 1
                continue
            if is_test:
                # A line that also appears in training is not a held-out test.
                if cleaned in train_seen:
                    dropped += 1
                    continue
                if cleaned in test_seen:
                    continue
                test_seen.add(cleaned)
                test_lines.append(cleaned)
            else:
                if cleaned in train_seen:
                    continue
                train_seen.add(cleaned)
                train_lines.append(cleaned)
            kept += 1

    for path, lines in ((train_path, train_lines), (test_path, test_lines)):
        with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
            for line in lines:
                f.write(line + '\n')
        size = sum(len(l.encode('utf-8')) for l in lines)
        print(f'  {path}: {len(lines)} lines, {size / 1e6:.1f} MB')

    print(f'Videos: {videos}. Kept {kept} captions, dropped {dropped}.')
    if not args.keep_raw and not args.local:
        os.remove(raw_path)
        print(f'  removed {raw_path}')

    # What the corpus is made of, so the split can be judged rather than
    # assumed.
    for name, path in (('train', train_path), ('test', test_path)):
        total = tamil = latin = 0
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                total += len(line)
                tamil += len(TAMIL.findall(line))
                latin += len(LATIN.findall(line))
        if total:
            print(f'  {name}: {100 * tamil / total:.1f}% Tamil script, '
                  f'{100 * latin / total:.1f}% Latin')
    return 0


if __name__ == '__main__':
    sys.exit(main())
