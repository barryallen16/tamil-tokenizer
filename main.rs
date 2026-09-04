use tamil_tokenizer::FastBPEEncoder;
use std::env;
use std::fs;
use std::io::{self, Write};
use serde::Deserialize;
use rustc_hash::FxHashMap;

#[derive(Deserialize)]
struct TokenizerData {
    vocab: std::collections::HashMap<String, u32>,
}

fn main() {
    println!("=== Tamil Fast Encoder/Decoder ===\n");

    let vocab_file = env::args().nth(1).unwrap_or_else(|| "output/tamil_tokenizer.json".to_string());

    println!("Loading vocabulary from {}...", vocab_file);
    let data = match fs::read_to_string(&vocab_file) {
        Ok(d) => d,
        Err(e) => { eprintln!("Failed to read vocab file: {}", e); return; }
    };

    let tokenizer_data: TokenizerData = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to parse vocab JSON: {}", e); return; }
    };

    let vocab: FxHashMap<String, u32> = tokenizer_data.vocab.into_iter().collect();

    println!("Loaded {} tokens\n", vocab.len());

    let encoder = match FastBPEEncoder::new(vocab) {
        Ok(e) => e,
        Err(e) => { eprintln!("Failed to create encoder: {}", e); return; }
    };
    
    // Interactive mode
    println!("Enter your text to encode (or 'quit' to exit):\n");
    
    loop {
        print!("> ");
        let _ = io::stdout().flush();
        
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() { break; }
        let input = input.trim();
        
        if input.is_empty() || input.eq_ignore_ascii_case("quit") {
            break;
        }
        
        // Encode
        let token_ids = encoder.encode(input);
        println!("\nEncoded to {} tokens:", token_ids.len());
        println!("  {:?}", token_ids);

        // Decode, dropping BOS/EOS so the result is comparable to the input
        let decoded = encoder.decode_skip_special(&token_ids);
        println!("\nDecoded:");
        println!("  {}", decoded);

        // Verify
        println!("\nMatch: {}\n", if input == decoded { "OK" } else { "MISMATCH" });
        println!("{}", "=".repeat(60));
        println!();
    }
    
    println!("Goodbye!");
}
