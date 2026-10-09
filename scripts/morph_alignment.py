#!/usr/bin/env python3
"""MorphScore for Tamil: how often our token boundaries land on morpheme
boundaries, against general-purpose tokenizers.

This is the axis `IMPROVEMENT_ROADMAP.md` 7.2 identifies as unmeasured, and the
one place the project's premise can be attacked from outside. Bytes/token says
our tokenizer is compact; it says nothing about whether the boundaries are
linguistically real.

Read the caveat at the bottom before treating a number here as a target. Arnett
& Bergen (COLING 2025) and Arnett, Hudspeth & O'Connor (ICML 2025) both find
morphological alignment does *not* explain model performance variance. So this
is a descriptive instrument, not a quality target.

Usage:
  pip install transformers
  git clone -b v2 https://github.com/catherinearnett/morphscore /tmp/morphscore
  python scripts/morph_alignment.py --our hf_export/v4_mixed_120000_prefixed/tokenizer.json
"""
import argparse
import os
import sys

MORPHSCORE_DIR = os.environ.get('MORPHSCORE_DIR', '/tmp/morphscore')

BASELINES = {
    'mBERT': 'bert-base-multilingual-cased',
    'Llama3': 'meta-llama/Meta-Llama-3-8B',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--our', required=True, help='path to our exported tokenizer.json')
    ap.add_argument('--skip', action='append', default=[], help='baselines to skip')
    ap.add_argument('--out', default='morphscore_results.json')
    args = ap.parse_args()

    if not os.path.exists(os.path.join(MORPHSCORE_DIR, 'morphscore.py')):
        sys.exit(f'morphscore not found at {MORPHSCORE_DIR}\n'
                 f'  git clone -b v2 https://github.com/catherinearnett/morphscore {MORPHSCORE_DIR}')

    sys.path.insert(0, MORPHSCORE_DIR)
    from morphscore import MorphScore
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer as HFCoreTokenizer

    class Shim:
        """Our export is a `tokenizers` BPE file; MorphScore was written against
        `transformers` tokenizers and needs `special_tokens_map`, a callable
        returning `input_ids`, and `decode`. A core `tokenizers.Tokenizer` has
        only the last, so supply the other two.

        `special_tokens_map` values are used to filter tokens out of the scored
        segmentation, so they must be the vocab's actual special strings — an
        empty dict would let `<|startoftext|>` through if it ever appeared."""

        def __init__(self, path):
            self.t = HFCoreTokenizer.from_file(path)
            self.special_tokens_map = {
                t: None
                for t in self.t.get_vocab()
                if t.startswith('<|') or t == ''
            }

        def __len__(self):
            return self.t.get_vocab_size()

        def convert_ids_to_tokens(self, ids):
            return [self.t.id_to_token(i) for i in ids]

        def decode(self, token_id):
            # MorphScore calls this both per-id and (in some versions) on a
            # batch, and expects the token's own surface string back.
            if isinstance(token_id, int):
                return self.t.id_to_token(token_id)
            return self.t.decode(list(token_id)).strip()

        def __call__(self, text):
            # MorphScore indexes token_output['input_ids'] rather than reading
            # `.tokens`, so return the dict-shaped result it expects.
            enc = self.t.encode(text)
            return {'input_ids': enc.ids, 'tokens': enc.tokens}

    # Tamil in MorphScore is UD_Tamil-TTB, ISO 639-3 tam + script taml.
    # data_dir defaults to a relative path, so point it at the clone or the
    # lookup fails with a bare "Dataset not found".
    scorer = MorphScore(language_subset=['tam_taml'], by_split=False,
                        freq_scale=True, exclude_single_tok=False,
                        data_dir=os.path.join(MORPHSCORE_DIR, 'clean_datasets'))

    targets = {'ours': Shim(args.our)}
    for name, repo in BASELINES.items():
        if name in args.skip:
            continue
        try:
            targets[name] = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            print(f'  skip {name}: {type(e).__name__}: {str(e)[:80]}')

    print(f'{"tokenizer":<12} {"recall":>9} {"precision":>10} {"tok/char":>9} {"n":>6}')
    results = {}
    for name, tok in targets.items():
        try:
            raw, df = scorer.eval(tok, return_df=True)
            # eval() returns {config, <lang>: {metrics}}; pull the Tamil row out.
            m = raw['tam_taml']
            results[name] = {
                'morphscore_recall': m['morphscore_recall'],
                'morphscore_precision': m['morphscore_precision'],
                'mean_token_char_ratio': m['mean_token_char_ratio'],
                'num_samples': m['num_samples'],
            }
            print(f"{name:<12} {m['morphscore_recall']:>9.4f} "
                  f"{m['morphscore_precision']:>10.4f} "
                  f"{m['mean_token_char_ratio']:>9.4f} {m['num_samples']:>6}")
            if name == 'ours':
                df.to_csv('morphscore_tamil_byitem.csv', index=False)
        except Exception as e:
            print(f'{name:<12} ERROR {type(e).__name__}: {str(e)[:100]}')

    import json
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {args.out}')
    print('\nMorphScore measures boundary alignment only. It is not a proxy for')
    print('downstream quality: the authors found alignment does not explain much')
    print('performance variance. Use it to describe our boundaries, not to rank')
    print('the tokenizer.')


if __name__ == '__main__':
    main()