#!/usr/bin/env python3
"""Evaluation metrics: compression ratio and fertility on Tamil, English, code."""
import json, os, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS_JSON = os.path.join(BASE, "output", "tamil_tokenizer_wiki.json")
RUST_BIN = os.path.join(BASE, "target", "release", "bench_encode.exe")

TAMIL = [
    "வணக்கம்! இது தமிழ் மொழியில் எழுதப்பட்ட ஒரு சோதனை வாக்கியம். இது மிகவும் நீண்ட வாக்கியம்.",
    "தமிழ்நாடு இந்தியாவின் தென் பகுதியில் அமைந்துள்ள ஒரு மாநிலமாகும். இதன் தலைநகர் சென்னை.",
    "செயற்கை நுண்ணறிவு மற்றும் இயந்திர கற்றல் தொழில்நுட்பங்கள் தமிழ் மொழியில் வளர்ந்து வருகின்றன.",
    "தமிழ் இலக்கியம் பல நூற்றாண்டுகள் பழமை வாய்ந்தது மற்றும் செழுமையானது.",
    "இந்த ஆய்வு தமிழ் மொழிக்கான சொல்லொட்டு பகுப்பாய்வை மேம்படுத்துகிறது.",
    "கணினி அறிவியல் மற்றும் தகவல் தொழில்நுட்பம் இன்று மிகவும் முக்கியமான துறைகளில் ஒன்று.",
    "தமிழ் மொழியில் இயற்கை வளங்கள் பற்றிய பல ஆய்வுகள் நடத்தப்பட்டுள்ளன.",
    "விளையாட்டு வீரர்கள் தமிழ்நாட்டின் பெருமையை உலக அரங்கில் நிலைநாட்டியுள்ளனர்.",
    "தமிழ் மொழியில் புதிய தொழில்நுட்பங்கள் பற்றிய ஆய்வுகள் அதிகரித்து வருகின்றன.",
    "இந்தியாவின் வரலாற்றில் தமிழர்களின் பங்களிப்பு மிகவும் முக்கியமானது.",
]

ENGLISH = [
    "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon.",
    "Artificial intelligence and machine learning are transforming how we process natural language.",
    "Byte pair encoding is a simple and effective tokenization algorithm used in modern language models.",
    "The transformer architecture has revolutionized natural language processing in recent years.",
    "Tokenization is the first and often most critical step in any NLP pipeline.",
    "Deep learning models require large amounts of labeled data for supervised training.",
    "The attention mechanism allows models to focus on relevant parts of the input sequence.",
    "Transfer learning enables models to leverage knowledge from pre-training on large corpora.",
    "Neural networks have achieved state-of-the-art performance on many language understanding tasks.",
    "The field of natural language processing continues to advance rapidly with new architectures.",
]

CODE = [
    "def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)",
    "for (int i = 0; i < n; i++) { sum += arr[i]; } return sum;",
    "SELECT * FROM users WHERE age > 18 AND country = 'India' ORDER BY name;",
    "class Node { int data; Node next; public Node(int val) { this.data = val; } }",
    "if (response.status_code == 200) { return response.json(); } else { throw new Error('Failed'); }",
    "import pandas as pd; df = pd.read_csv('data.csv'); print(df.head())",
    "try { const result = await fetch(url); const data = await result.json(); } catch (e) { console.error(e); }",
    "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE);",
    "while (queue.length > 0) { const node = queue.shift(); process(node); }",
    "git add . && git commit -m 'Initial commit' && git push origin main",
]


def encode_ours(text):
    r = subprocess.run([RUST_BIN, OURS_JSON, text], capture_output=True, cwd=BASE)
    output = r.stdout.decode("utf-8").strip()
    return output.split() if output else []


def compute_metrics(texts, name):
    total_bytes = total_tokens = total_words = 0
    for text in texts:
        tokens = encode_ours(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokens)
        total_words += len(text.split())
    return {
        "tokenizer": name,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0,
        "tokens_per_word": total_tokens / total_words if total_words else 0,
    }


def main():
    results = {}
    print("=" * 70)
    print("  EVALUATION METRICS -- Tamil Tokenizer")
    print("  Compression Ratio & Fertility Score")
    print("=" * 70)

    for lang, corpus in [("Tamil", TAMIL), ("English", ENGLISH), ("Code", CODE)]:
        m = compute_metrics(corpus, f"Ours ({lang})")
        results[lang] = m
        print(f"\n  [{lang}]")
        print(f"    Total bytes:      {m['total_bytes']:>8}")
        print(f"    Total tokens:     {m['total_tokens']:>8}")
        print(f"    Total words:      {m['total_words']:>8}")
        print(f"    Bytes/token:      {m['bytes_per_token']:>8.2f}")
        print(f"    Tokens/word:      {m['tokens_per_word']:>8.2f}")

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Language':<12} {'Bytes/Token':>12} {'Tokens/Word':>12} {'Tokens':>8}")
    print(f"  {'-' * 48}")
    for lang in ["Tamil", "English", "Code"]:
        m = results[lang]
        print(f"  {lang:<12} {m['bytes_per_token']:>12.2f} {m['tokens_per_word']:>12.2f} {m['total_tokens']:>8}")

    out_path = os.path.join(BASE, "output", "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
