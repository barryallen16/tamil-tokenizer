use tamil_tokenizer::{TamilTokenizer};
// use std::fs;

fn main() {
    let tokenizer = TamilTokenizer::new();
    tokenizer.print_stats();
    tokenizer.save_vocab("./output/tamil_vocab_cluster.json").unwrap();
}