use tamil_tokenizer::FastBPEEncoder;
use std::io::{self, BufRead};
use rustc_hash::FxHashMap;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <vocab_file> [repeats]", args[0]);
        std::process::exit(1);
    }

    let data = std::fs::read_to_string(&args[1])
        .unwrap_or_else(|e| { eprintln!("Failed to read vocab: {}", e); std::process::exit(1); });
    let tokenizer_data: serde_json::Value = serde_json::from_str(&data)
        .unwrap_or_else(|e| { eprintln!("Invalid JSON: {}", e); std::process::exit(1); });

    let vocab_map = if let Some(v) = tokenizer_data.get("vocab") {
        v.as_object().unwrap_or_else(|| { eprintln!("Missing vocab key"); std::process::exit(1); })
    } else {
        tokenizer_data.as_object().unwrap_or_else(|| { eprintln!("Invalid JSON format"); std::process::exit(1); })
    };

    let vocab: FxHashMap<String, u32> = vocab_map.iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
        .collect();

    let encoder = FastBPEEncoder::new_with_config(vocab, Default::default())
        .unwrap_or_else(|e| { eprintln!("Failed to create encoder: {}", e); std::process::exit(1); });

    let repeats: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let stdin = io::stdin();
    let lines: Vec<String> = stdin.lock().lines().map(|l| l.unwrap()).collect();

    for _ in 0..5 {
        for line in &lines {
            let _ = encoder.encode_ordinary(line);
        }
    }

    let start = Instant::now();
    let mut total_tokens = 0usize;
    for _ in 0..repeats {
        for line in &lines {
            let tokens = encoder.encode_ordinary(line);
            total_tokens += tokens.len();
        }
    }
    let elapsed = start.elapsed();

    let total_bytes: usize = lines.iter().map(|l| l.len()).sum();
    let throughput = (total_bytes as f64 * repeats as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
    let avg_per_doc = elapsed.as_secs_f64() / repeats as f64 / lines.len() as f64 * 1e6;

    let result = serde_json::json!({
        "throughput_mbps": throughput,
        "avg_per_doc_us": avg_per_doc,
        "total_tokens_per_iter": total_tokens / repeats,
        "total_bytes_per_iter": total_bytes,
        "repeats": repeats,
        "elapsed_ms": elapsed.as_millis(),
    });

    println!("{}", result);
}
