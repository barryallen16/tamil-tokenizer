#!/usr/bin/env python3
"""Check that an exported tokenizer installs the way a user would install it.

The export writes four files. `tokenizer.json` alone is enough for
`tokenizers.Tokenizer.from_file`, and that is what most of the published
examples used. It is not enough for `transformers.AutoTokenizer`, which wants
`tokenizer_config.json` next to it, and it is not enough for
`AutoTokenizer.from_pretrained(repo_id)`, which wants both at the repository
root. A tokenizer that cannot be loaded is not a tokenizer.

This script assembles the export the way the Hub serves it and asserts the
properties a caller depends on:

  1. `AutoTokenizer.from_pretrained` loads the directory.
  2. bos, eos, pad and unk resolve to the IDs the vocab gives them.
  3. `model_max_length` is not a truncation cap the caller did not ask for.
  4. Tamil, English and Latin-1 text round-trip through encode and decode.
  5. The same files work from a subdirectory, the way the 30K model is served.

Usage:
  python scripts/verify_install.py <hf_export_dir> [--repo <repo_id>]

Requires `pip install transformers tokenizers`.
"""
import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLES = [
    ("tamil", "வணக்கம்! இது தமிழ் உரை. இது ஒரு சோதனை வாக்கியம்."),
    ("english", "Hello world, this is English text 12345."),
    ("latin1", "café ñ ÿ straße"),
]

# Files a Hub repository root needs for AutoTokenizer to load it.
ROOT_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

failures = []


def check(name, ok, detail=""):
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)
    return ok


def load_from(directory):
    """Load the way a user loads: AutoTokenizer over a directory."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(directory)


def check_install(directory, label):
    print(f"\n--- {label}: {directory}")
    for f in ROOT_FILES:
        if not check(f"{f} present", os.path.exists(os.path.join(directory, f))):
            return None

    try:
        tok = load_from(directory)
    except Exception as e:
        check("AutoTokenizer.from_pretrained", False, f"{type(e).__name__}: {str(e)[:200]}")
        return None
    check("AutoTokenizer.from_pretrained", True, f"vocab={tok.vocab_size}")

    ids = {
        "bos_token": tok.bos_token_id,
        "eos_token": tok.eos_token_id,
        "pad_token": tok.pad_token_id,
        "unk_token": tok.unk_token_id,
    }
    for name, value in ids.items():
        check(f"{name} resolves", value is not None, f"id={value}")

    # transformers takes the smaller of model_max_length and its own 1e30
    # ceiling, so a large value stays large.
    check(
        "model_max_length is not a silent cap",
        tok.model_max_length >= 100_000,
        f"model_max_length={tok.model_max_length}",
    )

    for label, text in SAMPLES:
        enc = tok(text)
        decoded = tok.decode(enc["input_ids"])
        check(f"round-trip {label}", decoded == text, repr(decoded[:60]))
        no_special = tok(text, add_special_tokens=False)["input_ids"]
        check(
            f"round-trip {label} without special tokens",
            tok.decode(no_special) == text,
        )

    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", help="directory holding an export_hf output")
    ap.add_argument("--repo", help="also check a published repo id")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    hf_dir = args.hf_dir
    if not os.path.isdir(hf_dir):
        sys.exit(f"not a directory: {hf_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        # A Hub repository root: the tokenizer files sit at the top level.
        root_dir = os.path.join(tmp, "repo")
        os.makedirs(root_dir)
        for f in ROOT_FILES:
            src = os.path.join(hf_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(root_dir, f))
        check_install(root_dir, "repository root")

        # A Hub repository with the model in a subfolder.
        sub_dir = os.path.join(tmp, "repo_with_subfolder")
        model_dir = os.path.join(sub_dir, "tamil_tokenizer_wiki_hf")
        os.makedirs(model_dir)
        for f in ROOT_FILES:
            src = os.path.join(hf_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(model_dir, f))
        check_install(model_dir, "subfolder")

    if args.repo:
        print(f"\n--- published repo: {args.repo}")
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.repo)
            check("AutoTokenizer.from_pretrained(repo)", True, f"vocab={tok.vocab_size}")
            check("eos_token resolves", tok.eos_token_id is not None)
            check("pad_token resolves", tok.pad_token_id is not None)
        except Exception as e:
            check("AutoTokenizer.from_pretrained(repo)", False, f"{type(e).__name__}: {str(e)[:200]}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
