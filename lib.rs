pub mod vocab;
pub mod pretokenizer;
pub mod encoder;

pub use vocab::{TamilTokenizer, TokenizerError, Result};
pub use pretokenizer::TamilPretokenizer;
pub use encoder::FastBPEEncoder;