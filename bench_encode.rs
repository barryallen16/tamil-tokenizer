use tamil_tokenizer::{load_vocab, FastBPEEncoder};
use std::io::{self, BufRead};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <vocab_file> [text]", args[0]);
        std::process::exit(1);
    }

    let vocab = load_vocab(&args[1]).expect("Failed to load vocab");

    let encoder = FastBPEEncoder::new_with_config(vocab, Default::default())
        .expect("Failed to create encoder");

    if args.len() >= 3 {
        let text = &args[2];
        let tokens = encoder.encode_ordinary(text);
        let ids: Vec<String> = tokens.iter().map(|id| id.to_string()).collect();
        println!("{}", ids.join(" "));
    } else {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let text = line.expect("Failed to read line");
            let tokens = encoder.encode_ordinary(&text);
            let ids: Vec<String> = tokens.iter().map(|id| id.to_string()).collect();
            println!("{}", ids.join(" "));
        }
    }
}
