#!/usr/bin/env python3
"""Bits-per-byte + fertility for our tokenizer against public baselines.

Run on Colab T4 (16 GB). No training: bits-per-byte measures the *tokenizer*,
not the model, so one fixed small LM scoring the same text four ways gives a
paired, tokenizer-agnostic ranking. BPB = total NLL in nats / (ln2 * bytes).

READ THIS BEFORE INTERPRETING
----------------------------
Our Rust encoder and our HF export do NOT tokenize identically. Measured:
79.2% ID agreement, because FastBPEEncoder starts from 1,135 base clusters
while HF's BPE model starts from individual characters. Our encoder emits
longer tokens, so BPB measured on the export is *pessimistic* for us — it
scores the artifact an HF user gets, not the encoder we benchmark internally.

Run `score` locally on the same file and compare fertility. That gap is the
finding, not noise.

Usage (Colab):
    !python colab_bpb.py --our /content/tamil_tokenizer.json \\
                         --corpus /content/test.txt --mb 2

    # or compare several corpora
    !python colab_bpb.py --our /content/tamil_tokenizer.json \\
                         --corpus /content/wiki_test.txt --corpus /content/ic_test.txt
"""
import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tamil-capable, small enough for a T4. Qwen2.5 sees the most Tamil per
# parameter of the sub-2B options, which matters because a model blind to Tamil
# would rank tokenizers by noise.
DEFAULT_LM = "Qwen/Qwen2.5-0.5B"

BASELINES = {
    "mBERT": "bert-base-multilingual-cased",
    "Llama3": "meta-llama/Meta-Llama-3-8B",   # tokenizer only; gated on HF
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",          # tokenizer only
}


def load_tiktoken(enc):
    """cl100k_base via tiktoken. Not a `transformers` tokenizer, so it needs the
    same shim as ours. This is the baseline people actually ship, and unlike
    mBERT it is not 10 years old.

    tiktoken is BPE over byte pairs, so Tamil — which it has no vocabulary for
    — falls back to near one token per byte. Expect a tok/word in the high
    single digits. That is the finding, not a bug: it is what the tokenization
    tax looks like from the general-tokenizer end.
    """
    import tiktoken

    class Tik:
        def __init__(self, t):
            self.t = t

        def encode(self, text):
            return self.t.encode(text, allowed_special=set(), disallowed_special=())

        def id_to_piece(self, i):
            return None

    return Tik(tiktoken.get_encoding(enc))


TIKTOKEN_BASELINES = {"cl100k_base": "cl100k_base"}


def load_ours(path):
    """Our export is a `tokenizers` BPE file, not a HF transformers one."""
    from tokenizers import Tokenizer

    # A tokenizer.json parse failure on our own files is almost always the wrong
    # file: the repo publishes `tamil_tokenizer_wiki.json` in the trainer's
    # {vocab, merges, metadata} format, which is valid JSON but has no `model`
    # key. tokenizers reports that as a bare syntax error, so name it here.
    try:
        with open(path, encoding="utf-8") as f:
            head = json.load(f)
        if "model" not in head:
            keys = ", ".join(list(head)[:4])
            sys.exit(
                f"{path} is in the trainer's format ({{{keys}}}, ...), not an HF\n"
                f"tokenizer.json. Convert it first:\n"
                f"  cargo run --release --bin export_hf -- {path} hf_export/out 512\n"
                f"then pass hf_export/out/tokenizer.json instead.\n"
                f"Colab has no Rust toolchain, so upload an already-exported\n"
                f"tokenizer.json (see the repo's hf_export/ output)."
            )
    except json.JSONDecodeError:
        pass  # let tokenizers produce its own error below

    class Shim:
        """Minimal interface so the same code path serves ours and the baselines."""

        def __init__(self, t):
            self.t = t

        def encode(self, text):
            return self.t.encode(text).ids

        def id_to_piece(self, i):
            return self.t.id_to_token(i)

    return Shim(Tokenizer.from_file(path))


def read_corpus(paths, mb):
    limit = int(mb * 1024 * 1024)
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) < 20:
                    continue
                out.append(line)
                if sum(len(x) for x in out) >= limit:
                    break
    text = "\n".join(out)
    print(f"corpus: {len(text.encode('utf-8')) / 1e6:.2f} MB, {len(out):,} lines")
    return text


def fertility(tokenize, text, sample_lines=2000):
    """tokens/word, whitespace-split to match score.rs. Cheap, CPU only."""
    lines = text.split("\n")[:sample_lines]
    toks = sum(len(tokenize(l)) for l in lines)
    words = sum(len(l.split()) for l in lines)
    return toks / max(words, 1)


@torch.no_grad()
def bits_per_byte(tokenize, text, model, device, max_len=2048, block_chars=1 << 16):
    """Sum token NLL over the whole corpus, divide by ln(2) * bytes.

    The byte denominator is shared across tokenizers, which is the whole point.
    Never rebuild it from a rounded MB column.

    `block_chars` bounds the tokenize() call, not the model call: a byte-fallback
    tokenizer turns 64K chars into ~500K ids, so this is a memory guard, and it
    must be small. The model still sees `max_len` at a time.
    """
    n_bytes = len(text.encode("utf-8"))
    total_nats = 0.0
    total_tokens = 0

    for start in range(0, len(text), block_chars):
        ids = tokenize(text[start:start + block_chars])
        if not ids:
            continue
        for i in range(0, len(ids), max_len):
            window = ids[i:i + max_len]
            if len(window) < 2:
                continue
            t = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(t).logits.float()
            # next-token prediction: position j predicts token j+1
            nll = torch.nn.functional.cross_entropy(
                logits[0, :-1, :], t[0, 1:], reduction="sum"
            )
            total_nats += nll.item()
            total_tokens += len(window) - 1
        if start and start % (block_chars * 64) == 0:
            print(f"    ...{start / len(text):.0%}", flush=True)

    bpb = total_nats / (math.log(2) * n_bytes)
    return bpb, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", required=True, help="our exported tokenizer.json")
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--mb", type=float, default=2.0, help="MB per corpus")
    ap.add_argument("--lm", default=DEFAULT_LM, help=f"scoring LM (default {DEFAULT_LM})")
    ap.add_argument("--skip", action="append", default=[], help="baseline names to skip")
    ap.add_argument("--out", default="bpb_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", torch.cuda.get_device_name(0) if device == "cuda" else "")

    tokenizers = {"ours (HF export)": load_ours(args.our)}
    for name, repo in BASELINES.items():
        if name in args.skip:
            continue
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:90]}")
    for name, enc in TIKTOKEN_BASELINES.items():
        if name in args.skip:
            continue
        try:
            tokenizers[name] = load_tiktoken(enc)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:90]}")
    print(f"tokenizers: {list(tokenizers)}\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.lm, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    print(f"scoring LM: {args.lm} "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)\n")

    results = {}
    for path in args.corpus:
        text = read_corpus([path], args.mb)
        name = os.path.basename(path)
        print(f"\n=== {name} ===")

        # A tokenizer scored by the model that was TRAINED on it has a home
        # advantage no other row can match: per-token predictions are calibrated
        # to its own segmentation. That row is not a tokenizer comparison, it is
        # a tautology, and it will "win" almost regardless of Tamil quality.
        native = [t for t in tokenizers if t.lower().replace(".", "").replace(" ", "")
                  in args.lm.lower().replace("-", "")]
        if native:
            print(f"  WARNING: scoring LM's own tokenizer in the set ({native}).")
            print("  That row is not a fair comparison — see the note at the end.")

        print(f"{'tokenizer':<20} {'tok/word':>9} {'bits/byte':>10} {'tokens':>12}")

        results[name] = {}
        for tname, tk in tokenizers.items():
            fert = fertility(tk.encode, text)
            bpb, ntok = bits_per_byte(tk.encode, text, model, device)
            results[name][tname] = {"fertility": fert, "bpb": bpb, "tokens": ntok}
            print(f"{tname:<20} {fert:>9.3f} {bpb:>10.4f} {ntok:>12,}")

        best = min(results[name].items(), key=lambda kv: kv[1]['bpb'])
        ours = results[name]['ours (HF export)']['bpb']
        gap = 100 * (ours - best[1]['bpb']) / best[1]['bpb']
        print(f"\n  best: {best[0]}  |  ours is {gap:+.1f}% vs best (lower bpb is better)")

    with open(args.out, "w") as f:
        json.dump({"lm": args.lm, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nTwo caveats that decide whether these numbers mean anything:")
    print(" 1. Corpus. Raw MediaWiki dump is mostly ASCII markup (thumb|, {{cite")
    print("    web}}, HTML), which general tokenizers handle far better than a")
    print("    Tamil-specialised one. Score the CLEANED split, or ours is")
    print("    penalised for the corpus rather than measured on it.")
    print(" 2. Never compare against the scoring LM's own tokenizer. It was")
    print("    trained on that segmentation and gets a home advantage.")
    print(" 'ours' is the HF export, which segments ~1.7% worse than the Rust")
    print(" encoder (1.565 vs 1.539 tok/word on cleaned wiki held-out).")


if __name__ == "__main__":
    main()