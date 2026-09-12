pub mod encoder;
pub mod pretokenizer;
pub mod trainer;
pub mod vocab;

pub use encoder::FastBPEEncoder;
pub use pretokenizer::TamilPretokenizer;
pub use trainer::OptimizedBPETrainer;
pub use vocab::{Result as TokenizerResult, TamilTokenizer, TokenizerError};

/// A trained tokenizer file's contents: `token -> id` plus ordered merge pairs.
pub type TrainedTokenizer = (rustc_hash::FxHashMap<String, u32>, Vec<(String, String)>);

/// Load a vocab from either a trained tokenizer file (`{vocab, merges, metadata}`)
/// or a flat `{token: id}` map.
pub fn load_vocab(path: &str) -> TokenizerResult<rustc_hash::FxHashMap<String, u32>> {
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    json.get("vocab")
        .unwrap_or(&json)
        .as_object()
        .ok_or_else(|| TokenizerError::InvalidData(format!("{}: expected a JSON object", path)))?
        .iter()
        .map(|(k, v)| {
            v.as_u64().map(|id| (k.clone(), id as u32)).ok_or_else(|| {
                TokenizerError::InvalidData(format!("{}: id for {:?} is not an integer", path, k))
            })
        })
        .collect()
}

/// Load a full trained tokenizer file: vocab plus ordered merge pairs.
/// Returns an empty merge list for flat `{token: id}` maps.
pub fn load_tokenizer(path: &str) -> TokenizerResult<TrainedTokenizer> {
    let data = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&data)?;
    let vocab_obj = json
        .get("vocab")
        .unwrap_or(&json)
        .as_object()
        .ok_or_else(|| TokenizerError::InvalidData(format!("{}: expected a JSON object", path)))?;
    let mut vocab = rustc_hash::FxHashMap::default();
    for (k, v) in vocab_obj {
        let id = v.as_u64().ok_or_else(|| {
            TokenizerError::InvalidData(format!("{}: id for {:?} is not an integer", path, k))
        })?;
        vocab.insert(k.clone(), id as u32);
    }
    let mut merges = Vec::new();
    if let Some(list) = json.get("merges").and_then(|m| m.as_array()) {
        for pair in list {
            if let [a, b] = pair.as_array().map(Vec::as_slice).unwrap_or(&[]) {
                if let (Some(a), Some(b)) = (a.as_str(), b.as_str()) {
                    merges.push((a.to_string(), b.to_string()));
                }
            }
        }
    }
    Ok((vocab, merges))
}
