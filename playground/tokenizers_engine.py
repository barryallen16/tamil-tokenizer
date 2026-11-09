import json
import os
import sys
from functools import lru_cache

# Metadata for all supported tokenizers
AVAILABLE_MODELS = [
    {
        "id": "tamil-120k",
        "name": "Tamil Tokenizer (120K)",
        "category": "This Project",
        "hf_name": "barryallen16/tamil-tokenizer-wiki",
        "subfolder": None,
        "type": "hf",
        "description": "122K vocab BPE with pre-populated Tamil conjunct stems. Optimized on Wikipedia, IndicCorp, and transcripts."
    },
    {
        "id": "tamil-30k",
        "name": "Tamil Tokenizer (30K)",
        "category": "This Project",
        "hf_name": "barryallen16/tamil-tokenizer-wiki",
        "subfolder": "tamil_tokenizer_wiki_hf",
        "type": "hf",
        "description": "32K compact vocab BPE. Fast inference and small memory footprint."
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
        "id": "sarvam-2b",
        "name": "Sarvam AI (65K)",
        "category": "Indic LLMs",
        "hf_name": "sarvamai/sarvam-2b-v0.5",
        "subfolder": None,
        "type": "hf",
        "description": "65K Indian language foundation model tokenizer from Sarvam AI."
    },

    {
        "id": "mbert",
        "name": "mBERT (119K)",
        "category": "Baselines",
        "hf_name": "bert-base-multilingual-cased",
        "subfolder": None,
        "type": "hf",
        "description": "119K WordPiece cross-lingual baseline tokenizer."
    }
]

MODELS_BY_ID = {m["id"]: m for m in AVAILABLE_MODELS}
_LOADED_MODELS = {}


def get_model_instance(model_id: str):
    """Lazy load and cache tokenizer model instance."""
    if model_id in _LOADED_MODELS:
        return _LOADED_MODELS[model_id]

    meta = MODELS_BY_ID.get(model_id)
    if not meta:
        raise ValueError(f"Unknown model id: {model_id}")

    mtype = meta["type"]
    if mtype == "tiktoken":
        import tiktoken
        inst = tiktoken.get_encoding(meta["encoding_name"])
        _LOADED_MODELS[model_id] = inst
        return inst

    if mtype == "hf":
        from transformers import AutoTokenizer
        kwargs = {}
        if meta.get("subfolder"):
            kwargs["subfolder"] = meta["subfolder"]
        inst = AutoTokenizer.from_pretrained(meta["hf_name"], **kwargs)
        _LOADED_MODELS[model_id] = inst
        return inst

    raise ValueError(f"Unsupported model type: {mtype}")


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

            # Try to match offset in original string
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
        vocab_size = len(inst)
        backend = getattr(inst, "backend_tokenizer", None)

        if backend is not None:
            enc = backend.encode(text)
            token_ids = enc.ids
            offsets = enc.offsets

            for idx, (tid, (start, end)) in enumerate(zip(token_ids, offsets)):
                if start < end and end <= len(text):
                    substr = text[start:end]
                else:
                    substr = inst.decode([tid], skip_special_tokens=False)

                raw_bytes = substr.encode("utf-8")
                bytes_list = list(raw_bytes)
                hex_str = " ".join(f"0x{b:02X}" for b in raw_bytes)

                is_special = bool(
                    tid in inst.all_special_ids
                    if hasattr(inst, "all_special_ids")
                    else False
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
        else:
            token_ids = inst.encode(text, add_special_tokens=False)
            curr = 0
            for idx, tid in enumerate(token_ids):
                substr = inst.decode([tid], skip_special_tokens=False)
                raw_bytes = substr.encode("utf-8")
                tokens_out.append({
                    "index": idx,
                    "id": int(tid),
                    "text": substr,
                    "bytes": list(raw_bytes),
                    "hex": " ".join(f"0x{b:02X}" for b in raw_bytes),
                    "start": curr,
                    "end": curr + len(substr),
                    "is_special": False
                })
                curr += len(substr)

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
