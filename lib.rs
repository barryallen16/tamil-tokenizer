pub mod vocab;
pub mod pretokenizer;

pub use vocab::{TamilTokenizer, TokenizerError, Result};
pub use pretokenizer::TamilPretokenizer;