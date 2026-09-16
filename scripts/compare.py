#!/usr/bin/env python3
"""Head-to-head compression comparison on IDENTICAL bytes.

Every tokenizer encodes the same inputs: a corpus file plus fixed English and
code samples. Reports bytes/token and tokens/word per input class. No speed
claims: timing a 5 KB corpus measures warmup, not throughput.

Usage:
    uv pip install tiktoken sentencepiece tokenizers
    python scripts/compare.py [tokenizer.json] [corpus.txt]

tiktoken / sentencepiece / tokenizers are optional: missing ones are skipped.
HF and SentencePiece models are trained fresh on the given corpus so vocab
sizes stay roughly matched (1,000 and 500 here).
"""
import json
import os
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS_JSON = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(BASE, "output", "tamil_tokenizer.json")
CORPUS = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(BASE, "data", "tamil_corpus.txt")
RUST_BIN = os.path.join(BASE, "target", "release", "bench_encode" + (".exe" if os.name == "nt" else ""))
if not os.path.exists(RUST_BIN):
    sys.exit(f"Missing {RUST_BIN}\nRun: cargo build --release")

ENGLISH = ["The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon."] * 10 + [
    "Artificial intelligence and machine learning are transforming how we process natural language."] * 10
CODE = ["def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)"] * 10 + [
    "for (int i = 0; i < n; i++) { sum += arr[i]; } return sum;"] * 10


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def encode_ours(text):
    r = subprocess.run([RUST_BIN, OURS_JSON, text], capture_output=True, cwd=BASE)
    return r.stdout.decode("utf-8").split()


def ours_label():
    # Read the vocab size from the model rather than hardcoding it, so the
    # column stays honest when a different tokenizer.json is passed in.
    with open(OURS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    vocab = data.get("vocab", data)
    return f"ours-{len(vocab):,}"


def main():
    inputs = {"tamil": load_corpus(CORPUS), "english": ENGLISH, "code": CODE}
    results = {name: {} for name in inputs}
    label = ours_label()

    for name, texts in inputs.items():
        results[name][label] = [len(encode_ours(t)) for t in texts]

    try:
        import tiktoken
        tik = tiktoken.get_encoding("cl100k_base")
        for name, texts in inputs.items():
            results[name]["tiktoken"] = [len(tik.encode(t)) for t in texts]
    except ImportError:
        print("(skip tiktoken: not installed)")

    try:
        from tokenizers import Tokenizer, models, trainers
        hf = Tokenizer(models.BPE(unk_token="[UNK]"))
        hf.train([CORPUS], trainers.BpeTrainer(vocab_size=1000, special_tokens=["[UNK]"], min_frequency=1))
        for name, texts in inputs.items():
            results[name]["hf-bpe-1k"] = [len(hf.encode(t).ids) for t in texts]
    except ImportError:
        print("(skip hf tokenizers: not installed)")

    try:
        import sentencepiece as spm
        tmpdir = tempfile.mkdtemp()
        mp = os.path.join(tmpdir, "sp")
        spm.SentencePieceTrainer.train(input=CORPUS, model_prefix=mp, vocab_size=500,
                                       model_type="bpe", character_coverage=1.0,
                                       hard_vocab_limit=False)
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{mp}.model")
        for name, texts in inputs.items():
            results[name]["spm-bpe-500"] = [len(sp.encode_as_ids(t)) for t in texts]
    except ImportError:
        print("(skip sentencepiece: not installed)")

    print(f"{'input':<8} {'tokenizer':<12} {'bytes/tok':>9} {'tok/word':>9} {'tokens':>8}")
    for name, texts in inputs.items():
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        total_words = sum(len(t.split()) for t in texts)
        for tok, counts in results[name].items():
            n = sum(counts)
            print(f"{name:<8} {tok:<12} {total_bytes / n:>9.2f} {n / total_words:>9.2f} {n:>8}")

    out = os.path.join(BASE, "output", "comparison.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"\nCounts saved to {out}")


if __name__ == "__main__":
    main()
