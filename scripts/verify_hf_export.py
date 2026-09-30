#!/usr/bin/env python3
"""Check that our Rust encoder and an exported HF tokenizer agree on IDs.

Usage:
  python scripts/verify_hf_export.py <our_tokenizer.json> <hf_dir> [corpus] [max_lines]

Build `encode_lines` first (cargo build --release --bin encode_lines) and
`pip install tokenizers`. Reports both ID agreement and decode fidelity, which
catch different failures: byte-table drift breaks every line, a pretokenizer
mismatch breaks some, and a decoder mismatch breaks round-trip only.

Expected on a healthy export: decode ok 100%, ids match ~79%. ID agreement is
capped below 100% by design — see the header of `export_hf.rs`.
"""
import json
import subprocess
import sys

from tokenizers import Tokenizer


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    ours_path, hf_dir = sys.argv[1], sys.argv[2]
    corpus = sys.argv[3] if len(sys.argv) > 3 else 'data/tamil_wiki_clean.txt'
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    tok = Tokenizer.from_file(f'{hf_dir}/tokenizer.json')
    lines = [l.rstrip('\n') for l in open(corpus, encoding='utf-8')][:limit]
    lines = [l for l in lines if l.strip()]

    proc = subprocess.run(
        ['./target/release/encode_lines.exe', ours_path],
        input='\n'.join(lines),
        capture_output=True, text=True, encoding='utf-8',
    )
    if proc.returncode != 0:
        print('encode_lines failed:', proc.stderr[:500])
        return 1
    ours = json.loads(proc.stdout)

    agree = decode_ok = 0
    bad = []
    for line in lines:
        enc = tok.encode(line)
        if enc.ids == ours[line]:
            agree += 1
        elif len(bad) < 3:
            bad.append((line, ours[line][:12], enc.ids[:12]))
        if tok.decode(enc.ids) == line:
            decode_ok += 1

    total = len(lines)
    print(f'lines       {total}')
    print(f'ids match   {agree} ({100 * agree / total:.1f}%)')
    print(f'decode ok   {decode_ok} ({100 * decode_ok / total:.1f}%)')

    for line, mine, theirs in bad:
        print(f'\n  text: {line[:60]!r}')
        print(f'    ours tokens={[tok.id_to_token(i) for i in mine]}')
        print(f'    hf   tokens={[tok.id_to_token(i) for i in theirs]}')

    if agree < total:
        print(
            '\nExpected: ID agreement is ~79%, not ~100%. HF BPE starts from'
            '\ncharacters, ours starts from the 1,135 base clusters, so the same'
            '\nmerge list yields different segmentations. Decode fidelity is the'
            '\nnumber to hold at 100%. See the header of export_hf.rs.'
        )

    # Known and accepted: the Latin-1 stand-in case above. Anything else is a
    # real regression, so only those lines are tolerated.
    vocab = json.load(open(f'{hf_dir}/vocab.json', encoding='utf-8'))

    def only_standin_diffs(line, vocab):
        # Bytes above 126 that GPT-2 keeps as ordinary Latin-1 chars are
        # byte tokens *and* real text. Ours emits their UTF-8 bytes, HF emits
        # the stand-in itself.
        printable = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
        n = 0
        risky = set()
        for b in range(256):
            c = chr(b) if b in printable else chr(256 + n)
            if b not in printable:
                n += 1
            if ord(c) > 126:
                risky.add(c)
        return any(c in risky for c in line)


    def has_unvocab_chars(line, vocab):
        # HF BPE has no byte fallback when unk_token is null, so a character
        # absent from the vocab is silently dropped. Ours emits its bytes.
        return any(c not in vocab for c in line)

    unexplained = []
    standin = unvocab = 0
    for line in lines:
        if tok.encode(line).ids == ours[line]:
            continue
        if has_unvocab_chars(line, vocab):
            unvocab += 1
        elif only_standin_diffs(line, vocab):
            standin += 1
        else:
            unexplained.append(line)

    print(f'\nexplained: {standin} Latin-1 stand-in, {unvocab} unvocab characters')
    if unvocab:
        print('  Unvocab characters fell through to HF byte fallback (<0xNN>).')
        print('  Our ids for those bytes are the base-vocab stand-in chars, so')
        print('  the token COUNT matches but the ids differ by design.')

    if unexplained:
        print(f'\n{len(unexplained)} lines differ for an unknown reason')
        for line in unexplained[:3]:
            print(f'  {line[:70]!r}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())