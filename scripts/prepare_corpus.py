"""Unified corpus preparation pipeline for Tamil Tokenizer.

Handles:
  1. Wikipedia corpus cleaning (strips wikitext markup, templates, tags).
  2. IndicCorpV2 Tamil slice preparation (sentence splitting, prose filtering).
  3. YouTube video transcripts (downloads and filters conversational Tamil from sivabharani-comments).
  4. Splitting and mixing multi-domain corpora.

Usage:
  # Prepare Wikipedia
  python scripts/prepare_corpus.py --wiki data/tamil_wiki_clean.txt data/wiki

  # Prepare IndicCorp
  python scripts/prepare_corpus.py --indiccorp data/indiccorp_ta.txt data/indiccorp

  # Prepare YouTube video transcripts
  python scripts/prepare_corpus.py --transcripts data/transcripts

  # Mix corpora
  python scripts/prepare_corpus.py --mix data/mixed 200 wiki+ic+transcripts
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

DEFAULT_TRANSCRIPTS_URL = (
    "https://github.com/barryallen16/sivabharani-comments/raw/refs/heads/main/"
    "transcripts/output/tamil_transcripts.jsonl"
)

TAMIL = re.compile(r'[\u0B80-\u0BFF]')
LATIN = re.compile(r'[A-Za-z]')
ASCII_PUNCT = set(r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')
SENTENCE_SPLIT = re.compile(r'(?<=[.!?\u0964])\s+')


def is_tamil_prose(text: str) -> bool:
    text = text.strip()
    if len(text) < 10:
        return False
    letters = sum(1 for c in text if TAMIL.match(c))
    punct = sum(1 for c in text if c in ASCII_PUNCT)
    return letters >= len(text) * 0.4 and punct <= len(text) * 0.02


def clean_wikitext(text: str) -> str:
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return ' '.join(text.split())


def clean_caption(text: str) -> str:
    text = re.sub(r'\[[^\]]*\]|\([^\)]*\)', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = text.replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'")
    return ' '.join(text.split())


def download(url: str, dest: str) -> str:
    print(f"Downloading {url} -> {dest}")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, 'wb') as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print(f"  Downloaded {total / 1e6:.1f} MB")
    return dest


def prepare_transcripts(url: str, out_dir: str, local_file: str = None, test_share: float = 0.2):
    """Download and split conversational video transcripts from sivabharani-comments."""
    os.makedirs(out_dir, exist_ok=True)
    raw_path = local_file or os.path.join(out_dir, 'raw.jsonl')
    if not local_file:
        download(url, raw_path)

    train_path = os.path.join(out_dir, 'train.txt')
    test_path = os.path.join(out_dir, 'test.txt')
    train_lines, test_lines = [], []
    train_seen, test_seen = set(), set()
    videos = kept = dropped = 0

    with io.open(raw_path, encoding='utf-8') as f:
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
                videos += 1
                bucket = int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                is_test = bucket < test_share

                for item in items:
                    text = item.get('text', '') if isinstance(item, dict) else str(item)
                    cleaned = clean_caption(text)
                    if not is_tamil_prose(cleaned):
                        dropped += 1
                        continue

                    if is_test:
                        if cleaned in train_seen or cleaned in test_seen:
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
        size_mb = sum(len(l.encode('utf-8')) for l in lines) / 1e6
        print(f"  {path}: {len(lines)} lines, {size_mb:.1f} MB")

    print(f"Transcripts: {videos} videos. Kept {kept} lines, dropped {dropped}.")


def split_corpus(src: str, out_dir: str, holdout: int = 20000):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'train.txt')
    test_path = os.path.join(out_dir, 'test.txt')

    lines = [l.strip() for l in open(src, encoding='utf-8') if l.strip()]
    test, train = lines[-holdout:], lines[:-holdout]

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train) + '\n')
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test) + '\n')

    print(f"train {train_path} ({len(train)} lines)")
    print(f"test  {test_path} ({len(test)} lines)")


def mix_corpora(pairs, out_dir: str, sample_mb: int = None):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'train.txt')
    written = 0

    with open(out_path, 'w', encoding='utf-8') as out:
        for path, repeat in pairs:
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue
            lines = [l for l in open(path, encoding='utf-8') if l.strip()]
            if sample_mb and len(lines) > 1:
                cap = int(sample_mb * 1024 * 1024 / (os.path.getsize(path) / len(lines)))
                lines = lines[:cap]
            for _ in range(repeat):
                for l in lines:
                    written += len(l.encode('utf-8'))
                    out.write(l if l.endswith('\n') else l + '\n')
            print(f"  Added {path} x{repeat}: {len(lines)} lines")

    print(f"Mixed output -> {out_path} ({written / (1024 * 1024):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Corpus preparation for Tamil Tokenizer")
    parser.add_argument('--transcripts', action='store_true', help='Download and process video transcripts')
    parser.add_argument('--transcripts-url', default=DEFAULT_TRANSCRIPTS_URL, help='URL to transcripts JSONL')
    parser.add_argument('--transcripts-file', help='Path to local transcripts JSONL file')
    parser.add_argument('--split', nargs=2, metavar=('SRC', 'OUT_DIR'), help='Split corpus into train/test')
    parser.add_argument('--mix', nargs='+', help='Mix corpora: OUT_DIR [SAMPLE_MB] WHICH')
    args = parser.parse_args()

    if args.transcripts:
        out_dir = os.path.join(ROOT, 'data', 'transcripts')
        prepare_transcripts(args.transcripts_url, out_dir, args.transcripts_file)
        return

    if args.split:
        split_corpus(args.split[0], args.split[1])
        return

    if args.mix:
        out_dir = args.mix[0]
        sample = int(args.mix[1]) if len(args.mix) > 1 and args.mix[1].isdigit() else None
        which = args.mix[2] if len(args.mix) > 2 else 'wiki+ic+transcripts'
        sources = {
            'wiki': [('data/wiki/train.txt', 1)],
            'ic': [('data/indiccorp/train.txt', 1)],
            'wiki+ic': [('data/wiki/train.txt', 1), ('data/indiccorp/train.txt', 1)],
            'wiki+ic+transcripts': [
                ('data/wiki/train.txt', 1),
                ('data/indiccorp/train.txt', 1),
                ('data/transcripts/train.txt', 1),
            ],
            'wiki+transcripts': [
                ('data/wiki/train.txt', 1),
                ('data/transcripts/train.txt', 1),
            ],
        }.get(which, [('data/wiki/train.txt', 1)])
        mix_corpora(sources, out_dir, sample)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
