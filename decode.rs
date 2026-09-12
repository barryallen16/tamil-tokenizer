use std::env;
use tamil_tokenizer::{load_tokenizer, FastBPEEncoder};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <vocab_file> <token_ids...>", args[0]);
        eprintln!(
            "Example: {} ./output/tamil_tokenizer.json 245 1023 567",
            args[0]
        );
        std::process::exit(1);
    }

    let vocab_file = &args[1];
    let token_ids: Vec<u32> = args[2..].iter().filter_map(|s| s.parse().ok()).collect();

    if token_ids.is_empty() {
        eprintln!("Error: No valid token IDs provided");
        std::process::exit(1);
    }

    let (vocab, merges) = match load_tokenizer(vocab_file) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load vocab: {}", e);
            std::process::exit(1);
        }
    };

    let encoder = match FastBPEEncoder::new_with_merges(vocab, &merges) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create encoder: {}", e);
            std::process::exit(1);
        }
    };

    // Decode
    let decoded = encoder.decode(&token_ids);

    // Output
    println!("Token IDs: {:?}", token_ids);
    println!("Decoded: {}", decoded);
}
