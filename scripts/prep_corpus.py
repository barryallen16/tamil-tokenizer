#!/usr/bin/env python3
"""Turn an IndicCorpV2 slice into a training corpus plus a held-out split.

IndicCorp ships one document per line, 150-600 chars of news and web prose.
The trainer reads line by line, so this splits documents into sentences and
applies the same cleaning gate the wiki corpus uses, keeping the two training
sets comparable.

Usage:
  python scripts/prep_corpus.py data/indiccorp_ta.txt data/indiccorp
  python scripts/prep_corpus.py --split data/tamil_wiki_clean.txt data/wiki
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_wiki_fast import SENTENCE_SPLIT, clean_wikitext, is_tamil_prose


def prep(src, out_dir, holdout=50000):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'train.txt')
    test_path = os.path.join(out_dir, 'test.txt')

    docs = kept_sentences = dropped = 0
    test_lines = []

    with open(src, encoding='utf-8', errors='replace') as f, \
         open(train_path, 'w', encoding='utf-8') as train, \
         open(test_path, 'w', encoding='utf-8') as test:
        for raw in f:
            doc = clean_wikitext(raw.strip())
            if not doc:
                continue
            docs += 1
            for sentence in SENTENCE_SPLIT.split(doc):
                sentence = sentence.strip()
                if not is_tamil_prose(sentence):
                    if len(sentence) > 20:
                        dropped += 1
                    continue
                kept_sentences += 1
                # Hold out the tail of the file: IndicCorp is not shuffled, so
                # the end is the least likely to share topics with the head.
                if len(test_lines) < holdout:
                    test_lines.append(sentence)
                else:
                    train.write(sentence + '\n')
        for line in test_lines:
            test.write(line + '\n')

    def mb(p):
        return os.path.getsize(p) / (1024 * 1024)

    print(f'documents read     {docs}')
    print(f'sentences kept     {kept_sentences}')
    print(f'sentences dropped  {dropped}')
    print(f'train  {train_path}  {mb(train_path):.1f} MB')
    print(f'test   {test_path}  {mb(test_path):.1f} MB  ({len(test_lines)} lines)')
    return 0


def split(src, out_dir, holdout=20000):
    """Hold out the tail of a corpus that is already one sentence per line.

    Needed so a wiki-trained model is scored on wiki text it never saw;
    otherwise the "in-domain" number is just the training set.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, 'train.txt')
    test_path = os.path.join(out_dir, 'test.txt')

    lines = [l.strip() for l in open(src, encoding='utf-8') if l.strip()]
    test, train = lines[-holdout:], lines[:-holdout]

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train) + '\n')
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test) + '\n')

    def mb(p):
        return os.path.getsize(p) / (1024 * 1024)

    print(f'train  {train_path}  {mb(train_path):.1f} MB  ({len(train)} lines)')
    print(f'test   {test_path}  {mb(test_path):.1f} MB  ({len(test)} lines)')
    return 0


def mix(pairs, out_dir, sample_mb=None):
    """Concatenate training splits into one corpus.

    pairs is [(path, repeat_count)]. IndicCorp is ~15x the wiki volume, so
    sampling it keeps the wiki from being drowned out; pass repeat_count=1 for
    full volume.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'train.txt')

    written = 0
    kept = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for path, repeat in pairs:
            lines = [l for l in open(path, encoding='utf-8') if l.strip()]
            if sample_mb and len(lines) > 1:
                cap = int(sample_mb * 1024 * 1024 / (os.path.getsize(path) / len(lines)))
                lines = lines[:cap]
            for _ in range(repeat):
                for l in lines:
                    # Tamil is ~2.8 bytes per character, so len() on a str would
                    # undercount the file by that factor.
                    written += len(l.encode('utf-8'))
                    out.write(l)
                    kept += 1
            print(f'  {path} x{repeat}: {len(lines)} lines, '
                  f'{sum(len(l.encode("utf-8")) for l in lines) / (1024 * 1024):.1f} MB')

    print(f'mixed  {out_path}  {written / (1024 * 1024):.1f} MB  ({kept} lines)')
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['--mix']:
        out = sys.argv[2] if len(sys.argv) > 2 else 'data/mixed'
        sample = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
        which = sys.argv[4] if len(sys.argv) > 4 else 'wiki+ic'
        sources = {
            'wiki': [('data/wiki/train.txt', 1)],
            'ic': [('data/indiccorp/train.txt', 1)],
            'wiki+ic': [('data/wiki/train.txt', 1), ('data/indiccorp/train.txt', 1)],
        }[which]
        sys.exit(mix(sources, out, sample))
    if sys.argv[1:2] == ['--split']:
        src = sys.argv[2] if len(sys.argv) > 2 else 'data/tamil_wiki_clean.txt'
        out = sys.argv[3] if len(sys.argv) > 3 else 'data/wiki'
        sys.exit(split(src, out))
    src = sys.argv[1] if len(sys.argv) > 1 else 'data/indiccorp_ta.txt'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/indiccorp'
    sys.exit(prep(src, out))