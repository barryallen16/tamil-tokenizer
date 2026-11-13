#!/usr/bin/env python3
"""Colab Benchmark: Evaluates Partha against all 12 playground tokenizers on large foundation datasets.

Datasets:
  1. mozhi-ai/tamil-corpus (Curated multi-domain Tamil corpus)
  2. ai4bharat/sangraha (AI4Bharat Indic foundation pretraining corpus, synthetic/tam_Taml)

Usage:
  colab exec -s bench -f scripts/colab_benchmark.py
"""
import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

AVAILABLE_MODELS = [
    {
        "id": "tamil-120k",
        "name": "Partha Tamil (120K)",
        "category": "Partha",
        "hf_name": "barryallen16/partha-tamil-tokenizer",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "tamil-30k",
        "name": "Partha Tamil (30K)",
        "category": "Partha",
        "hf_name": "barryallen16/partha-tamil-tokenizer",
        "subfolder": "tamil_tokenizer_wiki_hf",
        "type": "hf",
    },
    {
        "id": "openai-o200k",
        "name": "OpenAI GPT-4o (o200k_base)",
        "category": "OpenAI",
        "encoding_name": "o200k_base",
        "type": "tiktoken",
    },
    {
        "id": "openai-cl100k",
        "name": "OpenAI GPT-4 / 3.5 (cl100k_base)",
        "category": "OpenAI",
        "encoding_name": "cl100k_base",
        "type": "tiktoken",
    },
    {
        "id": "gemma-4",
        "name": "Google Gemma 4 (262K)",
        "category": "Open LLMs",
        "hf_name": "google/gemma-4-E2B-it",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "qwen-3.8",
        "name": "Alibaba Qwen 3.8 (248K)",
        "category": "Open LLMs",
        "hf_name": "Qwen/Qwen3.8-27B",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "llama-4",
        "name": "Meta Llama 4 (201K)",
        "category": "Open LLMs",
        "hf_name": "unsloth/Llama-4-Scout-17B-16E-Instruct",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "muse-glimmer",
        "name": "Meta Muse Glimmer (202K)",
        "category": "Open LLMs",
        "hf_name": "meta-models/Muse-Glimmer-30B",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "indicbert",
        "name": "AI4Bharat IndicBERTv2 (256K)",
        "category": "Indic LLMs",
        "hf_name": "ai4bharat/IndicBERTv2-MLM-only",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "sarvam-105b",
        "name": "Sarvam AI 105B (262K)",
        "category": "Indic LLMs",
        "hf_name": "sarvamai/sarvam-105b",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "sarvam-1",
        "name": "Sarvam-1 (68K)",
        "category": "Indic LLMs",
        "hf_name": "sarvamai/sarvam-1",
        "subfolder": None,
        "type": "hf",
    },
    {
        "id": "mbert",
        "name": "mBERT (119K)",
        "category": "Baselines",
        "hf_name": "google-bert/bert-base-multilingual-cased",
        "subfolder": None,
        "type": "hf",
    },
]


def load_tokenizer(meta: dict):
    mtype = meta["type"]
    if mtype == "tiktoken":
        import tiktoken
        return tiktoken.get_encoding(meta["encoding_name"])

    if mtype == "hf":
        from tokenizers import Tokenizer
        if meta.get("subfolder"):
            from huggingface_hub import hf_hub_download
            fpath = hf_hub_download(meta["hf_name"], filename=f"{meta['subfolder']}/tokenizer.json")
            return Tokenizer.from_file(fpath)
        else:
            return Tokenizer.from_pretrained(meta["hf_name"])

    raise ValueError(f"Unknown type: {mtype}")


def fast_encode(inst, text: str) -> List[int]:
    if hasattr(inst, "encode") and not hasattr(inst, "get_vocab_size"):
        return inst.encode(text, allowed_special="all")
    enc = inst.encode(text)
    return enc.ids


def fetch_corpora() -> Dict[str, List[str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print("=" * 80)
    print("  DOWNLOADING REAL-WORLD FOUNDATION DATASETS FROM HUGGING FACE")
    print("=" * 80)

    corpora = {}

    # 1. mozhi-ai/tamil-corpus
    print("\n[1/2] Fetching mozhi-ai/tamil-corpus (data/2026-03-29.parquet)...")
    try:
        f_mozhi = hf_hub_download(
            repo_id="mozhi-ai/tamil-corpus",
            filename="data/2026-03-29.parquet",
            repo_type="dataset",
        )
        table_mozhi = pq.read_table(f_mozhi)
        col = "text" if "text" in table_mozhi.column_names else table_mozhi.column_names[0]
        lines_mozhi = []
        for val in table_mozhi[col].to_pylist():
            if val and isinstance(val, str):
                s = val.strip()
                if len(s) > 30:
                    lines_mozhi.append(s)
            if len(lines_mozhi) >= 2000:
                break
        corpora["mozhi_tamil_corpus"] = lines_mozhi
        bytes_mozhi = sum(len(x.encode("utf-8")) for x in lines_mozhi)
        print(f"  Loaded {len(lines_mozhi):,} documents ({bytes_mozhi:,} UTF-8 bytes)")
    except Exception as e:
        print(f"  Failed to load mozhi-ai/tamil-corpus: {e}")

    # 2. ai4bharat/sangraha
    print("\n[2/2] Fetching ai4bharat/sangraha (synthetic/tam_Taml/wiki_tam_Taml_0000_of_0063.parquet)...")
    try:
        f_sangraha = hf_hub_download(
            repo_id="ai4bharat/sangraha",
            filename="synthetic/tam_Taml/wiki_tam_Taml_0000_of_0063.parquet",
            repo_type="dataset",
        )
        table_sangraha = pq.read_table(f_sangraha)
        col = "text" if "text" in table_sangraha.column_names else table_sangraha.column_names[0]
        lines_sangraha = []
        for val in table_sangraha[col].to_pylist():
            if val and isinstance(val, str):
                s = val.strip()
                if len(s) > 30:
                    lines_sangraha.append(s)
            if len(lines_sangraha) >= 2000:
                break
        corpora["ai4bharat_sangraha"] = lines_sangraha
        bytes_sangraha = sum(len(x.encode("utf-8")) for x in lines_sangraha)
        print(f"  Loaded {len(lines_sangraha):,} documents ({bytes_sangraha:,} UTF-8 bytes)")
    except Exception as e:
        print(f"  Failed to load ai4bharat/sangraha: {e}")

    return corpora


def run_benchmark(corpus_name: str, lines: List[str]) -> List[dict]:
    total_bytes = sum(len(x.encode("utf-8")) for x in lines)
    total_words = sum(len(x.split()) for x in lines)

    print(f"\nEvaluating on '{corpus_name}': {len(lines):,} docs, {total_words:,} words, {total_bytes:,} bytes")

    results = []
    for meta in AVAILABLE_MODELS:
        mid = meta["id"]
        inst = load_tokenizer(meta)

        t0 = time.perf_counter()
        token_count = 0
        for line in lines:
            token_count += len(fast_encode(inst, line))
        elapsed = max(1e-5, time.perf_counter() - t0)

        # Free memory
        del inst
        gc.collect()

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

    # Savings vs GPT-4o
    o200k = next((r for r in results if r["model_id"] == "openai-o200k"), None)
    base_tok = o200k["token_count"] if o200k else results[0]["token_count"]
    for r in results:
        r["savings_vs_gpt4o_pct"] = round((1.0 - (r["token_count"] / base_tok)) * 100, 1)

    results.sort(key=lambda x: x["token_count"])
    return results


def print_table(title: str, results: List[dict]):
    print("\n" + "=" * 94)
    print(f"  {title.upper()}")
    print("=" * 94)
    print(f"  {'Rank':<5} {'Model':<30} {'Tokens':>10} {'Bytes/Tok':>10} {'Tok/Word':>9} {'Savings vs 4o':>14} {'KB/s':>9}")
    print("  " + "-" * 90)
    for rank, r in enumerate(results, 1):
        savings = f"{r['savings_vs_gpt4o_pct']:+.1f}%" if r["model_id"] != "openai-o200k" else "baseline"
        print(f"  {rank:<5} {r['name']:<30} {r['token_count']:>10,} {r['bytes_per_token']:>10.2f} {r['tokens_per_word']:>9.2f} {savings:>14} {r['throughput_kbs']:>9.1f}")
    print("=" * 94)


def main():
    corpora = fetch_corpora()
    all_results = {}

    for name, lines in corpora.items():
        res = run_benchmark(name, lines)
        all_results[name] = res
        print_table(f"Results: {name}", res)

    # Combined corpus
    combined_lines = []
    for lines in corpora.values():
        combined_lines.extend(lines)

    if combined_lines:
        comb_res = run_benchmark("combined_foundation_corpora", combined_lines)
        all_results["combined"] = comb_res
        print_table("OVERALL FOUNDATION BENCHMARK (mozhi-ai + sangraha)", comb_res)

    out_file = "/content/colab_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFinal results saved to {out_file}")


if __name__ == "__main__":
    main()
