#!/usr/bin/env python3
"""Comprehensive real-world benchmark comparing Partha against all 12 playground tokenizers.

Evaluates:
  - Wikipedia (formal encyclopedic prose)
  - YouTube transcripts (conversational Tamil speech from sivabharani-comments)
  - Thirukkural (classical Tamil literature)
  - Modern Tanglish & Tech (colloquial bilingual prose and code)

Usage:
  uv run python scripts/benchmark_playground.py [--force-fetch]
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from playground.tokenizers_engine import AVAILABLE_MODELS, get_model_instance

DATA_PATH = os.path.join(REPO_ROOT, "data", "realworld_benchmark.json")
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "playground_benchmark.json")

# Built-in fallbacks if offline
FALLBACK_THIRUKKURAL = [
    "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.",
    "கற்றதனால் ஆய பயனென்கொல் வாலறிவன் நற்றாள் தொழாஅர் எனின்.",
    "மலர்மிசை ஏகினான் மாணடி சேர்ந்தார் நிலமிசை நீடுவாழ் வார்.",
    "இருள்சேர் இருவினையும் சேரா இறைவன் பொருள்சேர் புகழ்புரிந்தார் மாட்டு.",
    "பொறிவாயில் ஐந்தவித்தான் பொய்தீர் ஒழுக்க நெறிநின்றார் நீடுவாழ் வார்.",
    "தனக்குவமை இல்லாதான் தாள்சேர்ந்தார்க் கல்லால் மனக்கவலை மாற்றல் அரிது.",
    "அறவாழி அந்தணன் தாள்சேர்ந்தார்க் கல்லால் பிறவாழி நீந்தல் அரிது.",
    "கோளில் பொறியின் குணமிலவே எண்குணத்தான் தாளை வணங்காத் தலை.",
    "பிறவிப் பெருங்கடல் நீந்துவர் நீந்தார் இறைவன் அடிசேரா தார்.",
    "துப்பார்க்குத் துப்பாய துப்பாக்கித் துப்பார்க்குத் துப்பாய தூஉம் மழை.",
]

FALLBACK_TANGLISH_TECH = [
    "வணக்கம் friends, today's video-ல we are going to learn modern LLM tokenization பற்றி.",
    "இந்த algorithm-ஐ Python-ல implement பண்ணும்போது memory consumption ரொம்ப முக்கியம்.",
    "Transformers model architecture-ல self-attention mechanism எப்படி work ஆகுதுன்னு பார்ப்போம்.",
    "நம்ம dataset-ல train பண்ணும்போது batch size 32 வச்சு gradient accumulation use பண்ணலாம்.",
    "React frontend கூட FastAPI backend connect பண்ணி REST API endpoints build பண்றோம்.",
    "Git repository clone பண்ணிட்டு `uv sync` run பண்ணா எல்லா dependencies-உம் install ஆயிடும்.",
    "Rust language-ல memory safety மற்றும் zero-cost abstractions ரொம்ப powerful-ஆ இருக்கும்.",
    "Database query optimize பண்ண indexing மற்றும் caching strategies கண்டிப்பா தேவை.",
    "Docker container deploy பண்ணும்போது environment variables ஒழுங்கா pass பண்ணனும்.",
    "Machine learning pipeline-ல data preprocessing step-தான் 80% time எடுத்துக்கும்.",
]


def fetch_realworld_data() -> Dict[str, List[str]]:
    """Fetch real-world datasets from Tamil Wikipedia, YouTube transcripts, and literature."""
    print("Fetching real-world evaluation corpora...")
    datasets: Dict[str, List[str]] = {
        "transcripts": [],
        "wikipedia": [],
        "literature": FALLBACK_THIRUKKURAL,
        "tanglish_tech": FALLBACK_TANGLISH_TECH,
    }

    # 1. Conversational YouTube transcripts
    transcripts_url = (
        "https://github.com/barryallen16/sivabharani-comments/raw/refs/heads/main/"
        "transcripts/output/tamil_transcripts.jsonl"
    )
    try:
        req = urllib.request.Request(transcripts_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            for _ in range(100):
                line = resp.readline().decode("utf-8").strip()
                if not line:
                    continue
                item = json.loads(line)
                for _vid, entries in item.items():
                    for entry in entries:
                        t = entry.get("text", "").strip()
                        if len(t) > 15:
                            datasets["transcripts"].append(t)
                        if len(datasets["transcripts"]) >= 250:
                            break
                    if len(datasets["transcripts"]) >= 250:
                        break
                if len(datasets["transcripts"]) >= 250:
                    break
        print(f"  - YouTube Transcripts: {len(datasets['transcripts'])} lines")
    except Exception as e:
        print(f"  - YouTube Transcripts fetch failed ({e}); using local sample")

    # 2. Formal Tamil Wikipedia articles
    wiki_topics = [
        "தமிழ்நாடு",
        "சென்னை",
        "செயற்கை_நுண்ணறிவு",
        "கணினி_நிரலாக்க_மொழி",
        "திருக்குறள்",
        "சூரியக்_குடும்பம்",
        "இயற்பியல்",
        "தமிழ்_இலக்கியம்",
        "இந்திய_வரலாறு",
        "மருத்துவம்",
    ]
    for topic in wiki_topics:
        encoded = urllib.parse.quote(topic)
        url = f"https://ta.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "TamilTokenizerBench/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                extract = data.get("extract")
                if extract and len(extract) > 40:
                    datasets["wikipedia"].append(extract)
        except Exception:
            pass
    print(f"  - Wikipedia Extracts: {len(datasets['wikipedia'])} articles")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)
    print(f"Corpora saved to {DATA_PATH}\n")
    return datasets


def load_realworld_data(force_fetch: bool = False) -> Dict[str, List[str]]:
    if not force_fetch and os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return fetch_realworld_data()


def encode_fast(model_inst, text: str) -> List[int]:
    """Fast encode without per-token object overhead."""
    # tiktoken
    if hasattr(model_inst, "encode") and not hasattr(model_inst, "get_vocab_size"):
        return model_inst.encode(text, allowed_special="all")
    # tokenizers.Tokenizer (HF)
    enc = model_inst.encode(text)
    return enc.ids


def evaluate_model_on_corpus(model_meta: dict, corpus: List[str]) -> dict:
    mid = model_meta["id"]
    inst = get_model_instance(mid)

    full_text = "\n".join(corpus)
    byte_count = len(full_text.encode("utf-8"))
    word_count = len([w for w in full_text.split() if w])

    t0 = time.perf_counter()
    tokens = []
    # Batch encode line by line to measure realistic throughput
    for line in corpus:
        tokens.extend(encode_fast(inst, line))
    elapsed = max(1e-5, time.perf_counter() - t0)

    token_count = len(tokens)
    bpt = round(byte_count / max(1, token_count), 2)
    tpw = round(token_count / max(1, word_count), 2)
    kbs = round((byte_count / 1024) / elapsed, 1)

    return {
        "model_id": mid,
        "name": model_meta["name"],
        "category": model_meta["category"],
        "token_count": token_count,
        "byte_count": byte_count,
        "word_count": word_count,
        "bytes_per_token": bpt,
        "tokens_per_word": tpw,
        "throughput_kbs": kbs,
        "time_seconds": round(elapsed, 3),
    }


def main():
    force_fetch = "--force-fetch" in sys.argv
    data = load_realworld_data(force_fetch=force_fetch)

    # Flatten combined corpus for overall rank
    all_lines = []
    for category, lines in data.items():
        all_lines.extend(lines)

    total_bytes = sum(len(x.encode("utf-8")) for x in all_lines)
    total_words = sum(len(x.split()) for x in all_lines)

    print("=" * 88)
    print(f"  REAL-WORLD TOKENIZER BENCHMARK -- ALL 12 PLAYGROUND MODELS")
    print(f"  Corpus: {len(all_lines)} lines, {total_words:,} words, {total_bytes:,} UTF-8 bytes")
    print(f"  Domains: Wikipedia ({len(data['wikipedia'])}), Transcripts ({len(data['transcripts'])}), "
          f"Literature ({len(data['literature'])}), Tanglish ({len(data['tanglish_tech'])})")
    print("=" * 88)

    results = []
    for model_meta in AVAILABLE_MODELS:
        res = evaluate_model_on_corpus(model_meta, all_lines)
        results.append(res)
        print(f"  Processed {model_meta['name']:<32} -> {res['token_count']:>7} tokens | {res['bytes_per_token']:>5.2f} B/tok | {res['tokens_per_word']:>5.2f} tok/w")

    # Find baseline (GPT-4o) for savings computation
    o200k_res = next((r for r in results if r["model_id"] == "openai-o200k"), None)
    base_tokens = o200k_res["token_count"] if o200k_res else results[0]["token_count"]

    for r in results:
        r["savings_vs_gpt4o_pct"] = round((1.0 - (r["token_count"] / base_tokens)) * 100, 1)

    # Sort by compression (tokens_per_word ascending / bytes_per_token descending)
    results.sort(key=lambda x: x["token_count"])

    print("\n" + "=" * 96)
    print("  FINAL REAL-WORLD COMPRESSION & FERTILITY RANKING (LOWER TOKENS = BETTER)")
    print("=" * 96)
    header = f"  {'Rank':<5} {'Model':<30} {'Tokens':>8} {'Bytes/Tok':>10} {'Tok/Word':>9} {'Savings vs 4o':>14} {'KB/s':>9}"
    print(header)
    print("  " + "-" * 92)

    for rank, r in enumerate(results, 1):
        savings_str = f"{r['savings_vs_gpt4o_pct']:+.1f}%" if r["model_id"] != "openai-o200k" else "baseline"
        print(f"  {rank:<5} {r['name']:<30} {r['token_count']:>8,} {r['bytes_per_token']:>10.2f} {r['tokens_per_word']:>9.2f} {savings_str:>14} {r['throughput_kbs']:>9.1f}")

    print("=" * 96)

    # Save structured results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_stats": {
                "total_lines": len(all_lines),
                "total_words": total_words,
                "total_bytes": total_bytes,
                "domains": {k: len(v) for k, v in data.items()}
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nStructured results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
