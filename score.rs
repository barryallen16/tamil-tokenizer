use std::time::Instant;
use tamil_tokenizer::{load_tokenizer, FastBPEEncoder};

/// Score a tokenizer on a corpus: bytes/token, tokens/word, throughput.
///
/// Reports throughput as a side effect so a compression run and a speed run do
/// not need separate passes. In-process, unlike a per-line subprocess, so a
/// 14 MB corpus scores in well under a second.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: score <tokenizer.json> <corpus> [label] [max_bytes]");
        std::process::exit(1);
    }

    let label = args.get(3).cloned().unwrap_or_else(|| args[2].clone());
    let max_bytes: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let (vocab, merges) = load_tokenizer(&args[1]).unwrap_or_else(|e| {
        eprintln!("load failed: {}", e);
        std::process::exit(1);
    });
    let enc = FastBPEEncoder::new_with_merges(vocab, &merges).unwrap_or_else(|e| {
        eprintln!("encoder failed: {}", e);
        std::process::exit(1);
    });

    let text = std::fs::read_to_string(&args[2]).unwrap_or_else(|e| {
        eprintln!("read failed: {}", e);
        std::process::exit(1);
    });

    let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut bytes = 0usize;
    let mut keep = 0usize;
    for l in &lines {
        if bytes + l.len() > max_bytes {
            break;
        }
        bytes += l.len();
        keep += 1;
    }
    lines.truncate(keep);

    for l in lines.iter().take(500) {
        let _ = enc.encode_ordinary(l);
    }

    let mut best = f64::MAX;
    let mut tokens = 0usize;
    let mut words = 0usize;
    for _ in 0..3 {
        let start = Instant::now();
        tokens = 0;
        words = 0;
        for l in &lines {
            tokens += enc.encode_ordinary(l).len();
            words += l.split_whitespace().count();
        }
        let mbps = bytes as f64 / (1024.0 * 1024.0) / start.elapsed().as_secs_f64();
        if mbps < best {
            best = mbps;
        }
    }

    let mb = bytes as f64 / (1024.0 * 1024.0);
    println!(
        "{}\t{:.1}\t{}\t{:.3}\t{:.3}\t{:.1}",
        label,
        mb,
        tokens,
        bytes as f64 / tokens as f64,
        tokens as f64 / words as f64,
        best
    );
}
