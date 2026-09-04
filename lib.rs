pub mod vocab;
pub mod pretokenizer;
pub mod trainer;
pub mod encoder;

pub use vocab::{TamilTokenizer, TokenizerError, Result as TokenizerResult};
pub use pretokenizer::TamilPretokenizer;
pub use trainer::OptimizedBPETrainer;
pub use encoder::FastBPEEncoder;

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
            v.as_u64()
                .map(|id| (k.clone(), id as u32))
                .ok_or_else(|| TokenizerError::InvalidData(format!("{}: id for {:?} is not an integer", path, k)))
        })
        .collect()
}
