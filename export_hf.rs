use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::io::Write;
use tamil_tokenizer::{TamilPretokenizer, TamilTokenizer, TokenizerResult as Result};

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
/// Known limit: under GPT-2's byte table, 162 byte-stand-in chars are also
/// ordinary Latin-1 text (`²`, `é`). Our encoder routes those through the byte
/// fallback and emits two tokens; HF, which sees raw text here, matches the
/// stand-in and emits one. It affects roughly 0.2% of Tamil lines. Matching it
/// would mean byte-mapping in a pretokenizer, which then breaks every raw-text
/// merge above.
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
    let (mut vocab, merges) = tamil_tokenizer::load_tokenizer(src)?;
    let trained_vocab = vocab.len();

    let (byte_encoder, byte_decoder) = TamilTokenizer::create_byte_mappings();
    let byte_chars: HashSet<char> = byte_decoder.keys().copied().collect();

    // A character used by any token but absent from the vocab cannot be looked
    // up by HF, so the input cannot even be represented. Append those.
    let mut missing: Vec<char> = vocab
        .keys()
        .flat_map(|t| t.chars())
        .filter(|c| !vocab.contains_key(&c.to_string()) && !byte_chars.contains(c))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    let added_prims = missing.len();
    let mut next_id = vocab.values().copied().max().map_or(0, |m| m + 1);
    for c in missing {
        vocab.insert(c.to_string(), next_id);
        next_id += 1;
    }

    // Byte fallback. Without it HF BPE silently drops any character absent from
    // the vocab, where our encoder emits its UTF-8 bytes. HF's convention is
    // `<0xNN>` tokens plus `byte_fallback: true`, and the decoder maps them
    // back through the same GPT-2 byte table.
    let mut byte_tokens = Vec::with_capacity(256);
    for b in 0..=255u8 {
        let name = format!("<0x{b:02X}>");
        vocab.insert(name.clone(), next_id);
        byte_tokens.push((name, byte_encoder[&b]));
        next_id += 1;
    }

    // Promotion: every multi-char token must be reachable from single
    // characters. Build each one left to right, and only emit a rule whose
    // result is itself a vocab token.
    let produced: HashSet<String> = merges.iter().map(|(a, b)| format!("{a}{b}")).collect();
    let mut reachable: HashSet<String> = byte_chars
        .iter()
        .map(|c| c.to_string())
        .chain(vocab.keys().filter(|t| t.chars().count() == 1).cloned())
        .collect();

    let mut candidates: Vec<&String> = vocab
        .keys()
        .filter(|t| t.chars().count() > 1 && !produced.contains(*t) && !reachable.contains(*t))
        .collect();
    // Longest result first. Promotion rank decides which merge BPE takes when
    // several are available, and our encoder resolves overlaps by longest
    // match. Emitting க்ஷெ (4 chars) before க்ஷ and ஷெ (3) is what makes HF
    // rebuild க்ஷெ instead of stopping at க் + ஷெ.
    candidates.sort_by_key(|t| (std::cmp::Reverse(t.chars().count()), (*t).clone()));

    let mut promotions: Vec<(String, String)> = Vec::new();
    for token in candidates {
        // Recursively split into two vocab tokens, longest prefix first. A
        // flat greedy pass would take the whole token as one part and emit
        // nothing; and a plain prefix pass cannot build ஸ்ரீ, since ஸ்ர is not
        // a token while ஸ் and ரீ both are.
        fn split(
            token: &str,
            vocab: &rustc_hash::FxHashMap<String, u32>,
            reachable: &HashSet<String>,
        ) -> Option<Vec<(String, String)>> {
            if reachable.contains(token) {
                return Some(Vec::new());
            }
            let chars: Vec<char> = token.chars().collect();
            for k in (1..chars.len()).rev() {
                let left: String = chars[..k].iter().collect();
                let right: String = chars[k..].iter().collect();
                if !vocab.contains_key(&left) || !vocab.contains_key(&right) {
                    continue;
                }
                let (Some(a), Some(b)) = (
                    split(&left, vocab, reachable),
                    split(&right, vocab, reachable),
                ) else {
                    continue;
                };
                let mut out = a;
                out.extend(b);
                out.push((left, right));
                return Some(out);
            }
            None
        }

        let Some(rules) = split(token, &vocab, &reachable) else {
            continue;
        };
        for (a, b) in rules {
            if reachable.insert(format!("{a}{b}")) {
                promotions.push((a, b));
            }
        }
    }

    // Promotions rank ahead of learned merges: a base cluster must exist before
    // any merge that consumes it.
    let mut all_merges = promotions.clone();
    all_merges.extend(merges.iter().cloned());

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

    // HF ships a ByteFallback decoder that turns <0xNN> back into the byte
    // char; ByteLevel then maps byte chars to raw bytes. One entry instead of
    // 256 Replace rules, and it cannot drift from the token names.
    let decoders = vec![
        json!({ "type": "ByteFallback" }),
        json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true,
        }),
    ];

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
    let mut config = Map::new();
    config.insert("tokenizer_class".into(), json!("PreTrainedTokenizerFast"));
    config.insert(
        "model_max_length".into(),
        json!(model_max_length.unwrap_or(512)),
    );
    config.insert("unk_token".into(), json!("<|unk|>"));
    config.insert("bos_token".into(), json!("<|startoftext|>"));
    config.insert("eos_token".into(), json!(""));
    write_file(
        &format!("{out_dir}/tokenizer_config.json"),
        &serde_json::to_string_pretty(&Value::Object(config))?,
    )?;

    println!(
        "Exported {} vocab, {} merges ({} promotions) to {}",
        trained_vocab,
        all_merges.len(),
        promotions.len(),
        out_dir
    );
    if added_prims > 0 {
        println!("  {added_prims} characters appended to the vocab as primitives");
    }
    println!("  tokenizer.json        (HF tokenizers, Split + BPE)");
    println!("  tokenizer_config.json (model_max_length, special tokens)");
    println!("  vocab.json            (slow-tokenizer vocab)");
    println!("  merges.txt            (slow-tokenizer merges, rank order)");
    Ok(())
}

fn write_file(path: &str, contents: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(contents.as_bytes())
}
