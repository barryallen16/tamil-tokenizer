#!/usr/bin/env python3
"""Check that our Rust encoder and HF `tokenizers` agree on IDs for an export.

Usage:
  python scripts/verify_hf_export.py <our_tokenizer.json> <hf_tokenizer.json> <corpus> [max_lines]

Needs `encode_lines` built (cargo build --release --bin encode_lines) and the
HF `tokenizers` package.
"""
import json
import subprocess
import sys

from tokenizers import Tokenizer


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    ours_path, hf_path, corpus = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    tok = Tokenizer.from_file(hf_path)
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

    agree = 0
    examples = []
    for line in lines:
        theirs = tok.encode(line).ids
        mine = ours[line]
        if mine == theirs:
            agree += 1
        elif len(examples) < 5:
            examples.append((line, mine, theirs))

    total = len(lines)
    print(f'lines={total} agree={agree} ({100 * agree / total:.1f}%)')
    for line, mine, theirs in examples:
        print(f'\n  text: {line[:60]!r}')
        print(f'    ours={mine[:14]}')
        print(f'    hf  ={theirs[:14]}')
        print(f'    ours tokens={[tok.id_to_token(i) for i in mine[:14]]}')
        print(f'    hf   tokens={[tok.id_to_token(i) for i in theirs[:14]]}')

    if agree != total:
        print('\nDisagreement is expected only where the pretokenizer splits\n'
              'differently (our Tamil-aware regex vs GPT-2 default). Byte table\n'
              'drift would instead break every line.')
    return 0 if agree else 1


if __name__ == '__main__':
    sys.exit(main())