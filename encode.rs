use tamil_tokenizer::{load_tokenizer, FastBPEEncoder};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <vocab_file> <text>", args[0]);
        std::process::exit(1);
    }
    
    let vocab_file = &args[1];
    let text = &args[2];
    
    let (vocab, merges) = match load_tokenizer(vocab_file) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to load vocab: {}", e); std::process::exit(1); }
    };

    let encoder = match FastBPEEncoder::new_with_merges(vocab, &merges) {
        Ok(e) => e,
        Err(e) => { eprintln!("Failed to create encoder: {}", e); std::process::exit(1); }
    };
    
    println!("Input: {}", text);
    let tokens = encoder.encode(text);
        let tokens1 = encoder.encode_ordinary(text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded (with special): {}", encoder.decode(&tokens));
    println!("Tokens without specials: {:?}", tokens1);    
    println!("Decoded (clean): {}", encoder.decode_skip_special(&tokens));
}
