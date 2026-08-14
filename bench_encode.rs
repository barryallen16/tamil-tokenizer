use tamil_tokenizer::FastBPEEncoder;
use std::io::{self, BufRead};
use rustc_hash::FxHashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <vocab_file> [text]", args[0]);
        std::process::exit(1);
    }

    let data = std::fs::read_to_string(&args[1]).expect("Failed to read vocab");
    let tokenizer_data: serde_json::Value = serde_json::from_str(&data).expect("Invalid JSON");

    let vocab_map = if let Some(v) = tokenizer_data.get("vocab") {
        v.as_object().expect("Missing vocab key").clone()
    } else {
        tokenizer_data.as_object().expect("Invalid JSON format").clone()
    };

    let vocab: FxHashMap<String, u32> = vocab_map.iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
        .collect();

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
