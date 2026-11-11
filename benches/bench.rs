use std::fs;
use std::time::Instant;
use tamil_tokenizer::{load_tokenizer, FastBPEEncoder};

/// Throughput on a real corpus: single encode vs rayon batch, and the
/// compression ratio the trained model actually reaches.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench <tokenizer.json> [corpus] [bytes]");
        std::process::exit(1);
    }

    let (vocab, merges) = load_tokenizer(&args[1]).unwrap_or_else(|e| {
        eprintln!("load failed: {}", e);
        std::process::exit(1);
    });
    let corpus = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("data/tamil_wiki_clean.txt");
    let want_bytes: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8 * 1024 * 1024);

    let enc = FastBPEEncoder::new_with_merges(vocab, &merges).unwrap_or_else(|e| {
        eprintln!("encoder failed: {}", e);
        std::process::exit(1);
    });

    let text = fs::read_to_string(corpus).unwrap_or_else(|e| {
        eprintln!("read {}: {}", corpus, e);
        std::process::exit(1);
    });
    let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut bytes = 0usize;
    let mut take = 0usize;
    for l in &lines {
        if bytes + l.len() > want_bytes {
            break;
        }
        bytes += l.len();
        take += 1;
    }
    lines.truncate(take);

    println!(
        "vocab={} corpus={} lines={} bytes={}",
        enc.vocab_size(),
        corpus,
        lines.len(),
        bytes
    );

    // Warm up: first call builds the rayon pool and touches every page.
    for l in lines.iter().take(200) {
        let _ = enc.encode_ordinary(l);
    }
    let _ = enc.encode_ordinary_batch(&lines[..200.min(lines.len())]);

    let mb = bytes as f64 / (1024.0 * 1024.0);

    let mut best_seq = f64::MAX;
    let mut tokens = 0usize;
    for _ in 0..3 {
        let start = Instant::now();
        tokens = 0;
        for l in &lines {
            tokens += enc.encode_ordinary(l).len();
        }
        let mbps = mb / start.elapsed().as_secs_f64();
        if mbps < best_seq {
            best_seq = mbps;
        }
    }

    let mut best_par = f64::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        let out = enc.encode_ordinary_batch(&lines);
        let mbps = mb / start.elapsed().as_secs_f64();
        if mbps < best_par {
            best_par = mbps;
        }
        assert_eq!(out.len(), lines.len());
    }

    println!("single-thread  {:.2} MB/s", best_seq);
    println!(
        "rayon batch   {:.2} MB/s  ({:.2}x)",
        best_par,
        best_par / best_seq
    );
    println!(
        "tokens={} bytes/token={:.3}",
        tokens,
        bytes as f64 / tokens as f64
    );
}
