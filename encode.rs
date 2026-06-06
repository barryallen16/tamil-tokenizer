use tamil_tokenizer::FastBPEEncoder;
use std::fs;
use rustc_hash::FxHashMap;
use serde_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct TokenizerData {
    vocab: std::collections::HashMap<String, u32>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <vocab_file> <text>", args[0]);
        std::process::exit(1);
    }
    
    let vocab_file = &args[1];
    let text = &args[2];
    
    // Load vocab
    let data = match fs::read_to_string(vocab_file) {
        Ok(d) => d,
        Err(e) => { eprintln!("Failed to read vocab file: {}", e); std::process::exit(1); }
    };
    let tokenizer_data: TokenizerData = match serde_json::from_str(&data) {
        Ok(d) => d,
        Err(e) => { eprintln!("Failed to parse vocab: {}", e); std::process::exit(1); }
    };
    
    let vocab: FxHashMap<String, u32> = tokenizer_data.vocab.into_iter().collect();
    
    let encoder = match FastBPEEncoder::new(vocab) {
        Ok(e) => e,
        Err(e) => { eprintln!("Failed to create encoder: {}", e); std::process::exit(1); }
    };
    
    println!("Input: {}", text);
    let tokens = encoder.encode(text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: {}", encoder.decode(&tokens));
}