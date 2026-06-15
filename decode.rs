use tamil_tokenizer::FastBPEEncoder;
use std::fs;
use std::env;
use serde_json;
use serde::Deserialize;
use rustc_hash::FxHashMap;

#[derive(Deserialize)]
struct TokenizerData {
    vocab: std::collections::HashMap<String, u32>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <vocab_file> <token_ids...>", args[0]);
        eprintln!("Example: {} ./output/tamil_tokenizer.json 245 1023 567", args[0]);
        std::process::exit(1);
    }

    let vocab_file = &args[1];
    let token_ids: Vec<u32> = args[2..]
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    if token_ids.is_empty() {
        eprintln!("Error: No valid token IDs provided");
        std::process::exit(1);
    }

    // Load vocabulary from tokenizer JSON
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

    // Decode
    let decoded = encoder.decode(&token_ids);

    // Output
    println!("Token IDs: {:?}", token_ids);
    println!("Decoded: {}", decoded);
}
