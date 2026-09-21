use serde_json::{json, Map, Value};
use std::io::Write;
use tamil_tokenizer::TokenizerResult as Result;

/// Write Hugging Face artifacts for a trained tokenizer.
///
/// Byte tokens already use GPT-2's byte-to-unicode table, so a `ByteLevel`
/// pretokenizer in Python produces the same IDs we do. The one thing that does
/// not translate is our pretokenizer regex: HF `tokenizers` has no field for
/// a custom pattern on `ByteLevel`, so the export uses the default GPT-2
/// regex. Chunk boundaries can differ from ours on whitespace; merge order and
/// IDs do not.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <tokenizer.json> <out_dir> [model_max_length]",
            args[0]
        );
        std::process::exit(1);
    }

    if let Err(e) = export(&args[1], &args[2], args.get(3).and_then(|s| s.parse().ok())) {
        eprintln!("Export failed: {}", e);
        std::process::exit(1);
    }
}

fn export(src: &str, out_dir: &str, model_max_length: Option<usize>) -> Result<()> {
    let (vocab, merges) = tamil_tokenizer::load_tokenizer(src)?;

    let vocab_size = vocab.len();
    let mut specials: Vec<(String, u32)> = vocab
        .iter()
        .filter(|(t, _)| t.starts_with("<|") || t.is_empty())
        .map(|(t, &i)| (t.clone(), i))
        .collect();
    specials.sort_by_key(|(_, id)| *id);

    // Serialization order is id order so the file is stable across runs.
    let mut by_id: Vec<(&String, &u32)> = vocab.iter().collect();
    by_id.sort_by_key(|(_, &id)| id);

    let mut model_vocab = Map::new();
    for (token, &id) in &by_id {
        model_vocab.insert((*token).clone(), json!(id));
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

    // merges.txt wants "a b" per line. Pair order is the rank.
    let merges_txt: String = merges
        .iter()
        .map(|(a, b)| format!("{} {}\n", a, b))
        .collect();

    let hf = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true,
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true,
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": Value::Object(model_vocab),
            "merges": merges
                .iter()
                .map(|(a, b)| vec![a.clone(), b.clone()])
                .collect::<Vec<_>>(),
        },
    });

    std::fs::create_dir_all(out_dir)?;

    let mut plain: Map<String, Value> = Map::new();
    for (token, &id) in &by_id {
        plain.insert((*token).clone(), json!(id));
    }

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
        &format!("#version: 0.2\n{merges_txt}"),
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
        "Exported {} vocab, {} merges to {}",
        vocab_size,
        merges.len(),
        out_dir
    );
    println!("  tokenizer.json        (HF tokenizers, ByteLevel + BPE)");
    println!("  tokenizer_config.json (model_max_length, special tokens)");
    println!("  vocab.json            (slow-tokenizer vocab)");
    println!("  merges.txt            (slow-tokenizer merges, rank order)");
    Ok(())
}

fn write_file(path: &str, contents: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(contents.as_bytes())
}
