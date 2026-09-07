use tamil_tokenizer::{load_tokenizer, FastBPEEncoder};
use std::env;
use std::io::{self, Write};

fn main() {
    println!("=== Tamil Fast Encoder/Decoder ===\n");

    let vocab_file = env::args().nth(1).unwrap_or_else(|| "output/tamil_tokenizer.json".to_string());

    println!("Loading vocabulary from {}...", vocab_file);
    let (vocab, merges) = match load_tokenizer(&vocab_file) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to load vocab: {}", e); return; }
    };

    println!("Loaded {} tokens\n", vocab.len());

    let encoder = match FastBPEEncoder::new_with_merges(vocab, &merges) {
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
