import collections
import gc
import json
import os
import sys
from functools import lru_cache

# Metadata for all supported tokenizers
AVAILABLE_MODELS = [
    {
        "id": "tamil-120k",
        "name": "Partha Tamil (120K)",
        "category": "Partha",
        "hf_name": "barryallen16/partha-tamil-tokenizer",
        "subfolder": None,
        "type": "hf",
        "description": "Partha 122K vocab BPE with pre-populated Tamil conjunct stems. Optimized on Wikipedia, IndicCorp, and transcripts."
    },
    {
        "id": "tamil-30k",
        "name": "Partha Tamil (30K)",
        "category": "Partha",
        "hf_name": "barryallen16/partha-tamil-tokenizer",
        "subfolder": "tamil_tokenizer_wiki_hf",
        "type": "hf",
        "description": "Partha 32K compact vocab BPE. Fast inference and small memory footprint."
    },
    {
        "id": "openai-o200k",
        "name": "OpenAI GPT-4o (o200k_base)",
        "category": "OpenAI",
        "encoding_name": "o200k_base",
        "type": "tiktoken",
        "description": "200K vocab BPE used in GPT-4o, GPT-4o-mini, and o1."
    },
    {
        "id": "openai-cl100k",
        "name": "OpenAI GPT-4 / 3.5 (cl100k_base)",
        "category": "OpenAI",
        "encoding_name": "cl100k_base",
        "type": "tiktoken",
        "description": "100K vocab BPE used in GPT-4 and GPT-3.5-turbo."
    },
    {
        "id": "gemma-4",
        "name": "Google Gemma 4 (262K)",
        "category": "Open LLMs",
        "hf_name": "google/gemma-4-E2B-it",
        "subfolder": None,
        "type": "hf",
        "description": "Google 262K open model tokenizer released in 2026."
    },
    {
        "id": "qwen-3.8",
        "name": "Alibaba Qwen 3.8 (248K)",
        "category": "Open LLMs",
        "hf_name": "Qwen/Qwen3.8-27B",
        "subfolder": None,
        "type": "hf",
        "description": "Alibaba 248K multilingual foundation model tokenizer released in 2026."
    },
    {
        "id": "llama-4",
        "name": "Meta Llama 4 (201K)",
        "category": "Open LLMs",
        "hf_name": "unsloth/Llama-4-Scout-17B-16E-Instruct",
        "subfolder": None,
        "type": "hf",
        "description": "Meta 201K vocabulary BPE tokenizer used in Llama 4 Scout and Maverick."
    },
    {
        "id": "muse-glimmer",
        "name": "Meta Muse Glimmer (202K)",
        "category": "Open LLMs",
        "hf_name": "meta-models/Muse-Glimmer-30B",
        "subfolder": None,
        "type": "hf",
        "description": "Meta's autumn 2026 foundation model 202K vocabulary BPE tokenizer."
    },
    {
        "id": "indicbert",
        "name": "AI4Bharat IndicBERTv2 (256K)",
        "category": "Indic LLMs",
        "hf_name": "ai4bharat/IndicBERTv2-MLM-only",
        "subfolder": None,
        "type": "hf",
        "description": "256K Indic language benchmark model developed by AI4Bharat."
    },
    {
        "id": "sarvam-105b",
        "name": "Sarvam AI 105B (262K)",
        "category": "Indic LLMs",
        "hf_name": "sarvamai/sarvam-105b",
        "subfolder": None,
        "type": "hf",
        "description": "262K vocabulary frontier Indic foundation model tokenizer from Sarvam AI."
    },
    {
        "id": "sarvam-1",
        "name": "Sarvam-1 (68K)",
        "category": "Indic LLMs",
        "hf_name": "sarvamai/sarvam-1",
        "subfolder": None,
        "type": "hf",
        "description": "68K Indian language foundation model tokenizer from Sarvam AI."
    },

    {
        "id": "mbert",
        "name": "mBERT (119K)",
        "category": "Baselines",
        "hf_name": "google-bert/bert-base-multilingual-cased",
        "subfolder": None,
        "type": "hf",
        "description": "119K WordPiece cross-lingual baseline tokenizer."
    }
]

MODELS_BY_ID = {m["id"]: m for m in AVAILABLE_MODELS}

# Strict LRU cache capped at 2 models to stay under 50 MB RAM on 512MB hosts
MAX_ACTIVE_MODELS = 2
_LOADED_MODELS = collections.OrderedDict()


def get_model_instance(model_id: str):
    """Lazy load and cache tokenizer model instance with strict memory eviction."""
    if model_id in _LOADED_MODELS:
        _LOADED_MODELS.move_to_end(model_id)
        return _LOADED_MODELS[model_id]

    meta = MODELS_BY_ID.get(model_id)
    if not meta:
        raise ValueError(f"Unknown model id: {model_id}")

    # Evict oldest model to keep memory footprint under 50MB
    while len(_LOADED_MODELS) >= MAX_ACTIVE_MODELS:
        _LOADED_MODELS.popitem(last=False)
        gc.collect()

    mtype = meta["type"]
    if mtype == "tiktoken":
        import tiktoken
        inst = tiktoken.get_encoding(meta["encoding_name"])
        _LOADED_MODELS[model_id] = inst
        return inst

    if mtype == "hf":
        from tokenizers import Tokenizer
        if meta.get("subfolder"):
            from huggingface_hub import hf_hub_download
            fpath = hf_hub_download(meta["hf_name"], filename=f"{meta['subfolder']}/tokenizer.json")
            inst = Tokenizer.from_file(fpath)
        else:
            inst = Tokenizer.from_pretrained(meta["hf_name"])
        _LOADED_MODELS[model_id] = inst
        return inst

    raise ValueError(f"Unsupported model type: {mtype}")


@lru_cache(maxsize=512)
def tokenize_text(model_id: str, text: str) -> dict:
    """Tokenize input text and return rich token metadata."""
    meta = MODELS_BY_ID.get(model_id)
    if not meta:
        raise ValueError(f"Unknown model id: {model_id}")

    inst = get_model_instance(model_id)
    mtype = meta["type"]

    tokens_out = []
    text_utf8 = text.encode("utf-8")
    char_count = len(text)
    byte_count = len(text_utf8)
    words = [w for w in text.split() if w]
    word_count = max(1, len(words))

    if mtype == "tiktoken":
        # Encode with tiktoken
        token_ids = inst.encode(text, allowed_special="all")
        vocab_size = inst.n_vocab

        curr_offset = 0
        for idx, tid in enumerate(token_ids):
            try:
                raw_bytes = inst.decode_single_token_bytes(tid)
            except KeyError:
                raw_bytes = b""
            bytes_list = list(raw_bytes)
            hex_str = " ".join(f"0x{b:02X}" for b in raw_bytes)
            decoded_text = raw_bytes.decode("utf-8", errors="replace")

            token_len_chars = len(decoded_text)
            start = curr_offset
            end = curr_offset + token_len_chars
            if text[start:end] == decoded_text:
                curr_offset = end
            else:
                found = text.find(decoded_text, curr_offset)
                if found != -1:
                    start = found
                    end = found + token_len_chars
                    curr_offset = end
                else:
                    start = curr_offset
                    end = curr_offset + token_len_chars

            tokens_out.append({
                "index": idx,
                "id": int(tid),
                "text": decoded_text,
                "bytes": bytes_list,
                "hex": hex_str,
                "start": start,
                "end": end,
                "is_special": False
            })

    elif mtype == "hf":
        vocab_size = inst.get_vocab_size()
        enc = inst.encode(text)
        token_ids = enc.ids
        offsets = enc.offsets
        tokens = enc.tokens

        for idx, (tid, (start, end), tok_str) in enumerate(zip(token_ids, offsets, tokens)):
            if start < end and end <= len(text):
                substr = text[start:end]
            else:
                dec = inst.decode([tid], skip_special_tokens=False)
                substr = tok_str if tok_str else dec

            raw_bytes = substr.encode("utf-8")
            bytes_list = list(raw_bytes)
            hex_str = " ".join(f"0x{b:02X}" for b in raw_bytes)

            is_special = bool(
                start == end or
                tok_str.startswith("<|") or
                tok_str.startswith("<s") or
                tok_str.startswith("</s") or
                tok_str.startswith("[") and tok_str.endswith("]")
            )

            tokens_out.append({
                "index": idx,
                "id": int(tid),
                "text": substr,
                "bytes": bytes_list,
                "hex": hex_str,
                "start": int(start),
                "end": int(end),
                "is_special": is_special
            })

    token_count = len(tokens_out)
    bytes_per_tok = round(byte_count / max(1, token_count), 2)
    toks_per_word = round(token_count / word_count, 2)

    return {
        "model_id": model_id,
        "model_name": meta["name"],
        "category": meta["category"],
        "description": meta["description"],
        "vocab_size": vocab_size,
        "token_count": token_count,
        "char_count": char_count,
        "byte_count": byte_count,
        "bytes_per_token": bytes_per_tok,
        "tokens_per_word": toks_per_word,
        "tokens": tokens_out
    }
