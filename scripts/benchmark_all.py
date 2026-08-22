#!/usr/bin/env python3
"""Benchmark: Tamil BPE vs tiktoken vs SentencePiece vs HuggingFace Tokenizers"""
import json, time, os, subprocess, tempfile, sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE, "data", "tamil_wiki_corpus.txt")
OURS_JSON = os.path.join(BASE, "output", "tamil_tokenizer_wiki.json")
RUST_SPEED = os.path.join(BASE, "target", "release", "bench_speed.exe")
RUST_ENCODE = os.path.join(BASE, "target", "release", "bench_encode.exe")

TAMIL = [
    "வணக்கம்! இது தமிழ் மொழியில் எழுதப்பட்ட ஒரு சோதனை வாக்கியம்.",
    "தமிழ்நாடு இந்தியாவின் தென் பகுதியில் அமைந்துள்ள ஒரு மாநிலமாகும்.",
    "செயற்கை நுண்ணறிவு மற்றும் இயந்திர கற்றல் தொழில்நுட்பங்கள் தமிழ் மொழியில் வளர்ந்து வருகின்றன.",
    "தமிழ் இலக்கியம் பல நூற்றாண்டுகள் பழமை வாய்ந்தது மற்றும் செழுமையானது.",
    "இந்த ஆய்வு தமிழ் மொழிக்கான சொல்லொட்டு பகுப்பாய்வை மேம்படுத்துகிறது.",
]
ENGLISH = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence and machine learning are transforming how we process natural language.",
    "Byte pair encoding is a simple and effective tokenization algorithm used in modern language models.",
    "The transformer architecture has revolutionized natural language processing in recent years.",
    "Tokenization is the first and often most critical step in any NLP pipeline.",
]
MIXED = [
    "Python is a popular programming language.",
    "OpenAI's GPT models use tiktoken for fast tokenization.",
    "Rust is blazingly fast for systems programming.",
]
ALL_TEXTS = {"Tamil": TAMIL, "English": ENGLISH, "Mixed": MIXED}


def encode_ours_one(text):
    r = subprocess.run([RUST_ENCODE, OURS_JSON, text], capture_output=True)
    return [int(x) for x in r.stdout.decode("utf-8").strip().split()]


def bench_ours_speed(texts, repeats=500):
    input_data = "\n".join(texts).encode("utf-8")
    r = subprocess.run([RUST_SPEED, OURS_JSON, str(repeats)], input=input_data, capture_output=True)
    return json.loads(r.stdout.strip())


def speed_test(encode_fn, texts, warmup=5, repeats=200):
    for t in texts[:warmup]: encode_fn(t)
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    start = time.perf_counter()
    for _ in range(repeats):
        for t in texts: encode_fn(t)
    elapsed = time.perf_counter() - start
    return {
        "throughput_mbps": (total_bytes * repeats) / elapsed / (1024 * 1024),
        "avg_per_doc_us": (elapsed / repeats) / len(texts) * 1e6,
    }


def measure(encode_fn, texts):
    all_tokens, all_bytes, all_words = [], 0, 0
    for t in texts:
        tokens = encode_fn(t)
        all_tokens.extend(tokens)
        all_bytes += len(t.encode("utf-8"))
        all_words += len(t.split())
    n = len(all_tokens)
    return {
        "total_tokens": n,
        "bytes_per_token": all_bytes / n if n else 0,
        "tokens_per_word": n / all_words if all_words else 0,
    }


def run_all(encode_fn, texts, vocab_size, speed_fn=None):
    comp = measure(encode_fn, texts)
    spd = speed_fn() if speed_fn else speed_test(encode_fn, texts)
    return {**comp, **spd, "vocab": vocab_size}


def main():
    results = {}

    print("[1/4] Our Tamil BPE (Rust)")
    with open(OURS_JSON, encoding="utf-8") as f:
        our_vocab_size = len(json.load(f)["vocab"])
    for lang, texts in ALL_TEXTS.items():
        r = run_all(encode_ours_one, texts, our_vocab_size,
                     speed_fn=lambda t=texts: bench_ours_speed(t, 500))
        results[f"Ours ({lang})"] = r
        print(f"  {lang}: {r['bytes_per_token']:.2f} B/tok, {r['throughput_mbps']:.1f} MB/s")

    print("[2/4] tiktoken (cl100k_base)")
    import tiktoken
    tik = tiktoken.get_encoding("cl100k_base")
    for lang, texts in ALL_TEXTS.items():
        enc = lambda t, _t=tik: _t.encode(t)
        r = run_all(enc, texts, tik.n_vocab)
        results[f"tiktoken ({lang})"] = r
        print(f"  {lang}: {r['bytes_per_token']:.2f} B/tok, {r['throughput_mbps']:.1f} MB/s")

    print("[3/4] SentencePiece BPE (30K)")
    tmpdir = tempfile.mkdtemp()
    mp = os.path.join(tmpdir, "sp")
    spm.SentencePieceTrainer.train(input=CORPUS, model_prefix=mp, vocab_size=30000,
                                    model_type="bpe", character_coverage=1.0)
    sp = spm.SentencePieceProcessor(); sp.Load(f"{mp}.model")
    for lang, texts in ALL_TEXTS.items():
        enc = lambda t, _sp=sp: _sp.encode_as_ids(t)
        r = run_all(enc, texts, sp.get_piece_size())
        results[f"SentencePiece ({lang})"] = r
        print(f"  {lang}: {r['bytes_per_token']:.2f} B/tok, {r['throughput_mbps']:.1f} MB/s")

    print("[4/4] HF Tokenizers BPE (30K)")
    hf = Tokenizer(models.BPE(unk_token="[UNK]"))
    hf.pre_tokenizer = pre_tokenizers.Whitespace()
    hf.train([CORPUS], trainers.BpeTrainer(vocab_size=30000,
              special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"], min_frequency=2))
    for lang, texts in ALL_TEXTS.items():
        enc = lambda t, _hf=hf: _hf.encode(t).ids
        r = run_all(enc, texts, hf.get_vocab_size())
        results[f"HF Tokenizers ({lang})"] = r
        print(f"  {lang}: {r['bytes_per_token']:.2f} B/tok, {r['throughput_mbps']:.1f} MB/s")

    out = os.path.join(BASE, "output", "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    for lang in ["Tamil", "English", "Mixed"]:
        print(f"\n{'='*85}\n  {lang} Text\n{'='*85}")
        print(f"  {'Tokenizer':<26} {'Vocab':>7} {'B/Tok':>7} {'Tok/Word':>9} {'Speed':>10}")
        print(f"  {'-'*65}")
        for key in sorted(results.keys()):
            if lang in key:
                r = results[key]
                print(f"  {key:<26} {r['vocab']:>7} {r['bytes_per_token']:>7.2f} {r['tokens_per_word']:>9.2f} {r['throughput_mbps']:>8.1f} MB/s")


if __name__ == "__main__":
    main()
