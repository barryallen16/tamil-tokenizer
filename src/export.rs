use crate::{TamilPretokenizer, TokenizerResult as Result};
use serde_json::{json, Map, Value};
use std::io::Write;

/// Export a trained tokenizer to Hugging Face format artifacts.
///
/// Produces:
/// - `tokenizer.json`: Hugging Face fast tokenizer layout with Regex Split pretokenizer and ByteFallback decoder.
/// - `tokenizer_config.json`: Special tokens and model max length metadata.
/// - `special_tokens_map.json`: Explicit mappings for BOS, EOS, UNK, PAD.
/// - `vocab.json`: Canonical token-to-id mapping.
/// - `merges.txt`: Ranked merge list.
pub fn export_hf(src: &str, out_dir: &str, model_max_length: Option<usize>) -> Result<()> {
    let (trained_vocab, merges) = crate::load_tokenizer(src)?;

    let canonical = crate::canonical_model(&trained_vocab, &merges);
    let vocab = canonical.vocab;
    let all_merges = canonical.merges;
    let trained_vocab_len = trained_vocab.len();

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

    const UNLIMITED: u64 = u64::MAX;
    let mut config = Map::new();
    config.insert("tokenizer_class".into(), json!("PreTrainedTokenizerFast"));
    config.insert(
        "model_max_length".into(),
        json!(model_max_length.map(|v| v as u64).unwrap_or(UNLIMITED)),
    );
    config.insert("unk_token".into(), json!("<|unk|>"));
    config.insert("bos_token".into(), json!("<|startoftext|>"));
    config.insert("eos_token".into(), json!("<|endoftext|>"));
    config.insert("pad_token".into(), json!("<|pad|>"));
    config.insert("clean_up_tokenization_spaces".into(), json!(false));
    write_file(
        &format!("{out_dir}/tokenizer_config.json"),
        &serde_json::to_string_pretty(&Value::Object(config))?,
    )?;

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
        trained_vocab_len + canonical.added_prims + 256,
        all_merges.len(),
        canonical.promotions,
        out_dir
    );

    Ok(())
}

fn write_file(path: &str, contents: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(contents.as_bytes())
}
