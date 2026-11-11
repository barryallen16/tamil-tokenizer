use std::time::Instant;
use tamil_tokenizer::{load_tokenizer, FastBPEEncoder, TamilPretokenizer};

/// Where encode time goes, with the counts that explain it.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (vocab, merges) = load_tokenizer(&args[1]).unwrap();
    let enc = FastBPEEncoder::new_with_merges(vocab, &merges).unwrap();
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let bytes: usize = lines.iter().map(|l| l.len()).sum();
    let mb = bytes as f64 / (1024.0 * 1024.0);

    let p = TamilPretokenizer::new();
    for _ in 0..3 {
        for l in lines.iter().take(200) {
            let _ = enc.encode_ordinary(l);
        }
    }

    // Stage 1: pretokenize only.
    let mut chunks_total = 0usize;
    let mut mut_hist = [0usize; 24];
    let mut best_p = f64::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        chunks_total = 0;
        for l in &lines {
            for c in p.pretokenize(l) {
                chunks_total += 1;
                // One merged token per chunk is the floor; the real count comes
                // from stage 2. Histogram by chunk byte length instead.
                let b = c.len().min(23);
                mut_hist[b] += 1;
            }
        }
        let v = mb / start.elapsed().as_secs_f64();
        if v < best_p {
            best_p = v;
        }
    }

    // Stage 2: everything. The difference is automaton + merges + emit.
    let mut best_full = f64::MAX;
    let mut tok_total = 0usize;
    for _ in 0..3 {
        let start = Instant::now();
        tok_total = 0;
        for l in &lines {
            tok_total += enc.encode_ordinary(l).len();
        }
        let v = mb / start.elapsed().as_secs_f64();
        if v < best_full {
            best_full = v;
        }
    }

    println!(
        "bytes={} chunks={} avg_chunk_bytes={:.1}",
        bytes,
        chunks_total,
        bytes as f64 / chunks_total as f64
    );
    println!(
        "emitted tokens={} bytes/token={:.3}",
        tok_total,
        bytes as f64 / tok_total as f64
    );
    println!(
        "tokens per chunk={:.2}",
        tok_total as f64 / chunks_total as f64
    );
    println!("pretokenize only {:.2} MB/s", best_p);
    println!("full encode     {:.2} MB/s", best_full);
    println!(
        "ms/MB: pretokenize {:.0}  encode-rest {:.0}",
        1000.0 / best_p,
        1000.0 / best_full - 1000.0 / best_p
    );
    print!("chunk length histogram: ");
    for (i, &c) in mut_hist.iter().enumerate() {
        if c > 0 {
            print!("{}:{} ", i, c);
        }
    }
    println!();
}
