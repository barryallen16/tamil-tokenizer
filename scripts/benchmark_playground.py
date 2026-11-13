"""Unified benchmark evaluating Partha against all 12 playground tokenizers.

Datasets:
  - Foundation mode (default): 21.0 MB from `ai4bharat/sangraha` and `mozhi-ai/tamil-corpus`.
  - Quick mode (--quick): Scraped Wikipedia, YouTube transcripts, literature, and Tanglish.

Usage:
  uv run python scripts/benchmark_playground.py
  uv run python scripts/benchmark_playground.py --quick
"""
import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from playground.tokenizers_engine import AVAILABLE_MODELS, get_model_instance

RESULTS_PATH = os.path.join(REPO_ROOT, "results", "playground_benchmark.json")


def fast_encode(inst, text: str) -> List[int]:
    """Fast encode without per-token dictionary allocation."""
    if hasattr(inst, "encode") and not hasattr(inst, "get_vocab_size"):
        return inst.encode(text, allowed_special="all")
    enc = inst.encode(text)
    return enc.ids


def fetch_foundation_corpora() -> Dict[str, List[str]]:
    """Fetch 21 MB foundation datasets from Hugging Face."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print("Fetching foundation datasets from Hugging Face...")
    corpora = {}

    # 1. mozhi-ai/tamil-corpus
    print("  [1/2] Downloading mozhi-ai/tamil-corpus...")
    try:
        f_mozhi = hf_hub_download(
            repo_id="mozhi-ai/tamil-corpus",
            filename="data/2026-03-29.parquet",
            repo_type="dataset",
        )
        table_mozhi = pq.read_table(f_mozhi)
        col = "text" if "text" in table_mozhi.column_names else table_mozhi.column_names[0]
        lines_mozhi = [
            v.strip() for v in table_mozhi[col].to_pylist()
            if isinstance(v, str) and len(v.strip()) > 30
        ][:2000]
        corpora["mozhi_tamil_corpus"] = lines_mozhi
        bytes_mozhi = sum(len(x.encode("utf-8")) for x in lines_mozhi)
        print(f"    Loaded {len(lines_mozhi):,} docs ({bytes_mozhi:,} UTF-8 bytes)")
    except Exception as e:
        print(f"    Failed to download mozhi-ai/tamil-corpus: {e}")

    # 2. ai4bharat/sangraha
    print("  [2/2] Downloading ai4bharat/sangraha...")
    try:
        f_sangraha = hf_hub_download(
            repo_id="ai4bharat/sangraha",
            filename="synthetic/tam_Taml/wiki_tam_Taml_0000_of_0063.parquet",
            repo_type="dataset",
        )
        table_sangraha = pq.read_table(f_sangraha)
        col = "text" if "text" in table_sangraha.column_names else table_sangraha.column_names[0]
        lines_sangraha = [
            v.strip() for v in table_sangraha[col].to_pylist()
            if isinstance(v, str) and len(v.strip()) > 30
        ][:2000]
        corpora["ai4bharat_sangraha"] = lines_sangraha
        bytes_sangraha = sum(len(x.encode("utf-8")) for x in lines_sangraha)
        print(f"    Loaded {len(lines_sangraha):,} docs ({bytes_sangraha:,} UTF-8 bytes)")
    except Exception as e:
        print(f"    Failed to download ai4bharat/sangraha: {e}")

    return corpora


def fetch_quick_corpora() -> Dict[str, List[str]]:
    """Fetch quick sample corpus."""
    import urllib.parse
    import urllib.request

    print("Fetching quick evaluation corpora...")
    lines = []
    # Fetch 10 Wikipedia extracts
    topics = ["தமிழ்நாடு", "சென்னை", "செயற்கை_நுண்ணறிவு", "கணினி_நிரலாக்க_மொழி", "திருக்குறள்"]
    for t in topics:
        encoded = urllib.parse.quote(t)
        url = f"https://ta.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TamilTokenizerBench/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("extract"):
                    lines.append(data["extract"])
        except Exception:
            pass

    fallback_samples = [
        "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.",
        "கற்றதனால் ஆய பயனென்கொல் வாலறிவன் நற்றாள் தொழாஅர் எனின்.",
        "வணக்கம் friends, today's video-ல we are going to learn modern LLM tokenization பற்றி.",
        "Transformers model architecture-ல self-attention mechanism எப்படி work ஆகுதுன்னு பார்ப்போம்.",
    ]
    lines.extend(fallback_samples)
    return {"quick_sample": lines}


def evaluate_corpus(corpus_name: str, lines: List[str]) -> List[dict]:
    total_bytes = sum(len(x.encode("utf-8")) for x in lines)
    total_words = sum(len(x.split()) for x in lines)

    print(f"\nEvaluating on '{corpus_name}': {len(lines):,} docs, {total_words:,} words, {total_bytes:,} bytes")

    results = []
    for meta in AVAILABLE_MODELS:
        mid = meta["id"]
        inst = get_model_instance(mid)

        t0 = time.perf_counter()
        token_count = sum(len(fast_encode(inst, line)) for line in lines)
        elapsed = max(1e-5, time.perf_counter() - t0)

        bpt = round(total_bytes / max(1, token_count), 2)
        tpw = round(token_count / max(1, total_words), 2)
        kbs = round((total_bytes / 1024) / elapsed, 1)

        results.append({
            "model_id": mid,
            "name": meta["name"],
            "category": meta["category"],
            "token_count": token_count,
            "bytes_per_token": bpt,
            "tokens_per_word": tpw,
            "throughput_kbs": kbs,
            "elapsed_s": round(elapsed, 2),
        })

    # Relative savings vs first OpenAI baseline
    o200k = next((r for r in results if r["model_id"] == "openai-o200k"), None)
    base_tok = o200k["token_count"] if o200k else results[0]["token_count"]
    for r in results:
        r["savings_vs_openai_pct"] = round((1.0 - (r["token_count"] / base_tok)) * 100, 1)

    results.sort(key=lambda x: x["token_count"])
    return results


def print_table(title: str, results: List[dict]):
    print("\n" + "=" * 94)
    print(f"  {title.upper()}")
    print("=" * 94)
    print(f"  {'Rank':<5} {'Model':<34} {'Tokens':>10} {'Bytes/Tok':>10} {'Tok/Word':>9} {'Savings':>11} {'KB/s':>9}")
    print("  " + "-" * 90)
    for rank, r in enumerate(results, 1):
        savings = f"{r['savings_vs_openai_pct']:+.1f}%" if r["model_id"] != "openai-o200k" else "baseline"
        print(f"  {rank:<5} {r['name']:<34} {r['token_count']:>10,} {r['bytes_per_token']:>10.2f} {r['tokens_per_word']:>9.2f} {savings:>11} {r['throughput_kbs']:>9.1f}")
    print("=" * 94)


def main():
    parser = argparse.ArgumentParser(description="Tokenizer benchmark across playground models")
    parser.add_argument("--quick", action="store_true", help="Run quick sample instead of 21MB foundation corpora")
    args = parser.parse_args()

    corpora = fetch_quick_corpora() if args.quick else fetch_foundation_corpora()
    all_results = {}

    for name, lines in corpora.items():
        res = evaluate_corpus(name, lines)
        all_results[name] = res
        print_table(name, res)

    combined_lines = [line for lines in corpora.values() for line in lines]
    if len(corpora) > 1 and combined_lines:
        comb_res = evaluate_corpus("combined_foundation_corpora", combined_lines)
        all_results["combined"] = comb_res
        print_table("OVERALL FOUNDATION BENCHMARK (mozhi-ai + sangraha)", comb_res)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
