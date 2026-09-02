use tamil_tokenizer::FastBPEEncoder;
use std::fs;
use std::time::Instant;
use serde::Deserialize;
use serde_json;
use rustc_hash::FxHashMap;

#[derive(Deserialize)]
struct TokenizerData {
    vocab: std::collections::HashMap<String, u32>,
}

fn format_duration(nanos: u128) -> String {
    if nanos < 1_000 {
        format!("{} ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.2} us", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", nanos as f64 / 1_000_000_000.0)
    }
}

fn format_throughput(bytes: usize, nanos: u128) -> String {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    let secs = nanos as f64 / 1_000_000_000.0;
    format!("{:.2} MB/s", mb / secs)
}

fn load_corpus(filepath: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(filepath)?;
    // Split by newlines (assuming one text per line)
    // Or customize parsing based on your corpus format
    Ok(content
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn main() {
    println!("=== Tamil Tokenizer Corpus Benchmark ===\n");
    
    // Load vocabulary
    let vocab_file = "./output/tamil_tokenizer.json";
    let data = match fs::read_to_string(vocab_file) {
        Ok(d) => d,
        Err(e) => { eprintln!("Failed to read vocab file '{}': {}", vocab_file, e); std::process::exit(1); }
    };
    let tokenizer_data: TokenizerData = match serde_json::from_str(&data) {
        Ok(d) => d,
        Err(e) => { eprintln!("Failed to parse vocab: {}", e); std::process::exit(1); }
    };
    
    let vocab: FxHashMap<String, u32> = tokenizer_data.vocab.into_iter().collect();
    println!("Vocabulary size: {}\n", vocab.len());
    
    let start = Instant::now();
    let encoder = match FastBPEEncoder::new(vocab) {
        Ok(e) => e,
        Err(e) => { eprintln!("Failed to create encoder: {}", e); std::process::exit(1); }
    };
    let build_time = start.elapsed();
    println!("Encoder build time: {}\n", format_duration(build_time.as_nanos()));
    
    // Load corpus
    let corpus_path = "./data/tamil_corpus.txt"; // Adjust path to your corpus
    let corpus_texts = match load_corpus(corpus_path) {
        Ok(t) => t,
        Err(e) => { eprintln!("Failed to load corpus '{}': {}", corpus_path, e); std::process::exit(1); }
    };
    
    println!("Loaded {} documents from corpus\n", corpus_texts.len());
    
    // Corpus-based benchmarking
    println!("Doc ID |    Chars |       Tokens |   Time  Throughput");
    let mut total_bytes = 0;
    let mut total_tokens = 0;
    let mut total_time_nanos = 0u128;
    
    let sample_size = corpus_texts.len().min(100); // Benchmark first 100 docs
    
    for (idx, text) in corpus_texts.iter().take(sample_size).enumerate() {
        let char_count = text.chars().count();
        let byte_count = text.len();
        
        // Warmup
        for _ in 0..3 {
            let _ = encoder.encode(text);
        }
        
        // Benchmark single encoding
        let start = Instant::now();
        let tokens = encoder.encode(text);
        let encode_time = start.elapsed();
        
        total_bytes += byte_count;
        total_tokens += tokens.len();
        total_time_nanos += encode_time.as_nanos();
        
        let throughput = format_throughput(byte_count, encode_time.as_nanos());
        
        println!("{:>6} | {:>7} | {:>12} | {:>8} {:>11}",
                 idx + 1, char_count, tokens.len(),
                 format_duration(encode_time.as_nanos()), throughput);
    }

    // Summary statistics
    println!("\n=== Corpus Summary ===");
    println!("Documents processed: {}", sample_size);
    println!("Total bytes: {}", total_bytes);
    println!("Total tokens: {}", total_tokens);
    println!("Average compression ratio: {:.2} bytes/token", 
             total_bytes as f64 / total_tokens as f64);
    println!("Overall throughput: {}", format_throughput(total_bytes, total_time_nanos));
    println!("Average time per document: {}", 
             format_duration(total_time_nanos / sample_size as u128));
    
    // Batch processing test
    println!("\n=== Batch Processing (all corpus) ===");
    let start = Instant::now();
    let _results: Vec<Vec<u32>> = corpus_texts.iter()
        .map(|t| encoder.encode(t))
        .collect();
    let batch_time = start.elapsed();
    
    println!("Batch size: {} documents", corpus_texts.len());
    println!("Total time: {}", format_duration(batch_time.as_nanos()));
    println!("Per document: {}", format_duration(batch_time.as_nanos() / corpus_texts.len() as u128));
}
