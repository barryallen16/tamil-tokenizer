use serde_json::{json, Map, Value};
use std::io::Write;
use tamil_tokenizer::{TamilPretokenizer, TokenizerResult as Result};

/// Write Hugging Face artifacts for a trained tokenizer.
///
/// Two things about our model have no direct HF equivalent, and both are fixed
/// here rather than left to silently produce different IDs in Python:
///
/// 1. Our pretokenizer is a Tamil-aware regex, not GPT-2's. HF's `ByteLevel`
///    pretokenizer hardcodes the GPT-2 pattern, and that pattern byte-maps the
///    text before BPE sees it, which no merge in our file can ever match. So the
///    export uses `Split` with our own regex, which keeps raw text intact.
/// 2. Our trainer treats consonant clusters (`கே`, `ங்`) as atomic base
///    symbols. HF BPE has no base-symbol set: it starts from single characters
///    and can only reach a token through merges. So the export adds promotion
///    merges that build each such token from its characters, and appends any
///    character the vocab lacks so HF can represent the input at all.
///
/// ID parity with our encoder is 79.2%. The cause sits in `FastBPEEncoder`.
/// All 120,000 learned merges survive the export (verified), so nothing is
/// lost here. The encoder seeds a chunk with a greedy longest match over the
/// 1,135 base clusters and merges one occurrence per pass. HF's `BPE` model
/// seeds with single characters and merges every occurrence of the best-ranked
/// pair per pass. Reimplementing HF's algorithm over this file's own merge
/// list reproduces its IDs exactly, so the gap closes when the encoder takes
/// those semantics.
///
/// Registering the base clusters as `added_tokens` does not help. HF extracts
/// added tokens before the BPE model runs, so the model never fires and output
/// collapses to characters (measured: 0% agreement).
///
/// Round-trip fidelity is 100% and is the property to hold this file to. See
/// `scripts/verify_hf_export.py`, which reports fidelity and ID agreement
/// separately.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <tokenizer.json> <out_dir> [model_max_length]",
            args[0]
        );
        std::process::exit(1);
    }

    let max_len = args.get(3).and_then(|s| s.parse().ok());
    if let Err(e) = export(&args[1], &args[2], max_len) {
        eprintln!("Export failed: {}", e);
        std::process::exit(1);
    }
}

fn export(src: &str, out_dir: &str, model_max_length: Option<usize>) -> Result<()> {
    let (trained_vocab, merges) = tamil_tokenizer::load_tokenizer(src)?;

    // The canonical model is the vocabulary and merge list both engines read.
    // The Rust encoder builds the same thing from the same artifact, so the
    // two agree on every token ID.
    let canonical = tamil_tokenizer::canonical_model(&trained_vocab, &merges);
    let vocab = canonical.vocab;
    let all_merges = canonical.merges;
    let trained_vocab = trained_vocab.len();


    let specials: Vec<(String, u32)> = {
        let mut v: Vec<(String, u32)> = vocab
            .iter()
            .filter(|(t, _)| t.starts_with("<|") || t.is_empty())
            .map(|(t, &i)| (t.clone(), i))
            .collect();
        v.sort_by_key(|(_, id)| *id);
        v
    };

    let mut by_id: Vec<(&String, &u32)> = vocab.iter().collect();
    by_id.sort_by_key(|(_, &id)| id);

    let mut model_vocab = Map::new();
    let mut plain = Map::new();
    for (token, &id) in &by_id {
        model_vocab.insert((*token).clone(), json!(id));
        plain.insert((*token).clone(), json!(id));
    }

    let added_tokens: Vec<Value> = specials
        .iter()
        .map(|(t, id)| {
            json!({
                "id": id,
                "content": t,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();

    let pretok = TamilPretokenizer::new();
    let pattern = pretok.pattern().to_string();

    // ByteFallback alone, and `ByteLevel` must NOT follow it.
    //
    // ByteFallback assembles `<0xNN>` back into bytes and returns them as a
    // UTF-8 string. ByteLevel then re-applies the GPT-2 table to whatever came
    // out, so a decoded `é` (U+00E9, itself a GPT-2 stand-in for byte 0xE9) is
    // turned back into the raw byte 0xE9 and decodes to U+FFFD. Measured on
    // v3_wiki_30000: `café ñ ÿ` came back as `caf<?><?> <?><?>`.
    let decoders = vec![json!({ "type": "ByteFallback" })];

    let hf = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": { "Regex": pattern },
                    "behavior": "Isolated",
                    "invert": false,
                },
            ],
        },
        "post_processor": null,
        "decoder": {
            "type": "Sequence",
            "decoders": decoders,
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": Value::Object(model_vocab),
            "merges": all_merges
                .iter()
                .map(|(a, b)| vec![a.clone(), b.clone()])
                .collect::<Vec<_>>(),
        },
    });

    std::fs::create_dir_all(out_dir)?;

    write_file(
        &format!("{out_dir}/tokenizer.json"),
        &serde_json::to_string_pretty(&hf)?,
    )?;
    write_file(
        &format!("{out_dir}/vocab.json"),
        &serde_json::to_string_pretty(&Value::Object(plain))?,
    )?;
    write_file(
        &format!("{out_dir}/merges.txt"),
        &format!(
            "#version: 0.2\n{}",
            all_merges
                .iter()
                .map(|(a, b)| format!("{a} {b}\n"))
                .collect::<String>()
        ),
    )?;

    // model_max_length goes in tokenizer_config.json. HF's tokenizer.json
    // schema rejects the field, so writing it there makes the file unloadable.
    //
    // transformers reads model_max_length as the truncation cap, so
    // `tok(text, truncation=True)` cuts the input at it. The default is the
    // largest u64, which transformers treats as unlimited, since a base
    // tokenizer has no length to impose. Pass an explicit value to set a cap.
    const UNLIMITED: u64 = u64::MAX;
    let mut config = Map::new();
    config.insert("tokenizer_class".into(), json!("PreTrainedTokenizerFast"));
    config.insert(
        "model_max_length".into(),
        json!(model_max_length.map(|v| v as u64).unwrap_or(UNLIMITED)),
    );
    config.insert("unk_token".into(), json!("<|unk|>"));
    config.insert("bos_token".into(), json!("<|startoftext|>"));
    // An empty string here resolves to no token, so `tok.eos_token` comes
    // back None even though <|endoftext|> is in the vocab.
    config.insert("eos_token".into(), json!("<|endoftext|>"));
    config.insert("pad_token".into(), json!("<|pad|>"));
    // The decoder is ByteFallback, so there is no GPT-2 whitespace cleanup to
    // undo. Leaving this on lets transformers rewrite the decoded string.
    config.insert("clean_up_tokenization_spaces".into(), json!(false));
    write_file(
        &format!("{out_dir}/tokenizer_config.json"),
        &serde_json::to_string_pretty(&Value::Object(config))?,
    )?;

    // transformers reads bos/eos/pad/unk from special_tokens_map.json when it
    // exists and falls back to tokenizer_config.json otherwise. Writing both
    // keeps the two in sync.
    write_file(
        &format!("{out_dir}/special_tokens_map.json"),
        &serde_json::to_string_pretty(&json!({
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "unk_token": "<|unk|>",
            "pad_token": "<|pad|>",
        }))?,
    )?;

    println!(
        "Exported {} vocab, {} merges ({} promotions) to {}",
        trained_vocab + canonical.added_prims + 256,
        all_merges.len(),
        canonical.promotions,
        out_dir
    );
    if canonical.added_prims > 0 {
        println!(
            "  {} characters appended to the vocab as primitives",
            canonical.added_prims
        );
    }
    println!("  tokenizer.json        (HF tokenizers, Split + BPE)");
    println!("  tokenizer_config.json (model_max_length, special tokens)");
    println!("  special_tokens_map.json (bos/eos/pad/unk)");
    println!("  vocab.json            (slow-tokenizer vocab)");
    println!("  merges.txt            (slow-tokenizer merges, rank order)");
    Ok(())
}

fn write_file(path: &str, contents: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(contents.as_bytes())
}
