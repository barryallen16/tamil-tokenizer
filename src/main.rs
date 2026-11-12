use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;
use tamil_tokenizer::{
    export_hf, load_tokenizer, FastBPEEncoder, OptimizedBPETrainer, TamilTokenizer,
};

fn print_usage(prog: &str) {
    eprintln!(
        r#"Tamil Tokenizer CLI

Usage:
  {prog} <command> [arguments]

Commands:
  train    <corpus> <num_merges> <output_json> [min_freq]   Train BPE tokenizer on a corpus
  encode   <vocab_json> [text] [--lines-json]                Encode text to token IDs
  decode   <vocab_json> [token_id ...]                       Decode token IDs back to text
  export   <vocab_json> <out_dir> [max_length]               Export Hugging Face tokenizer artifacts
  score    <vocab_json> <corpus> [label] [max_bytes]         Score compression and encoding throughput
  repl     [vocab_json]                                      Interactive REPL shell
"#
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let prog = args
        .first()
        .map(String::as_str)
        .unwrap_or("tamil_tokenizer");

    if args.len() < 2 {
        print_usage(prog);
        std::process::exit(1);
    }

    let cmd = args[1].as_str();
    let cmd_args = &args[2..];

    let result = match cmd {
        "train" => cmd_train(cmd_args),
        "encode" => cmd_encode(cmd_args),
        "decode" => cmd_decode(cmd_args),
        "export" => cmd_export(cmd_args),
        "score" => cmd_score(cmd_args),
        "repl" => cmd_repl(cmd_args),
        "help" | "-h" | "--help" => {
            print_usage(prog);
            Ok(())
        }
        unknown => {
            eprintln!("Unknown command: '{}'\n", unknown);
            print_usage(prog);
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_train(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 3 {
        eprintln!("Usage: train <corpus_file> <num_merges> <output_json> [min_frequency]");
        std::process::exit(1);
    }

    let corpus_file = &args[0];
    let num_merges: usize = args[1].parse()?;
    let output_file = &args[2];
    let min_frequency: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("=== Training Tamil Tokenizer ===");
    println!("Corpus: {}", corpus_file);
    println!("Merges: {}", num_merges);
    println!("Min frequency: {}", min_frequency);
    println!("Output: {}\n", output_file);

    let tokenizer = TamilTokenizer::new();
    let vocab_ahash: ahash::AHashMap<String, u32> = tokenizer.vocab.into_iter().collect();

    let mut trainer = OptimizedBPETrainer::new(vocab_ahash);
    trainer.set_verbose(true);
    trainer.set_min_frequency(min_frequency);

    println!("Loading corpus...");
    trainer.initialize_from_corpus(corpus_file)?;

    println!("Training BPE...");
    trainer.train(num_merges);

    println!("\nSaving tokenizer artifact...");
    trainer.save_tokenizer_with_reserved_tokens(output_file)?;

    println!("Vocabulary saved to {}", output_file);
    Ok(())
}

fn cmd_encode(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: encode <vocab_json> [text] [--lines-json]");
        std::process::exit(1);
    }

    let vocab_file = &args[0];
    let is_json_lines = args.iter().any(|a| a == "--lines-json");
    let text_arg = args.get(1).filter(|s| *s != "--lines-json");

    let (vocab, merges) = load_tokenizer(vocab_file)?;
    let enc = FastBPEEncoder::new_with_merges(vocab, &merges)?;

    if is_json_lines {
        let stdin = io::stdin();
        let mut map = serde_json::Map::new();
        for line in stdin.lock().lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            map.insert(line.clone(), serde_json::json!(enc.encode_ordinary(&line)));
        }
        println!("{}", serde_json::to_string(&map)?);
        return Ok(());
    }

    if let Some(text) = text_arg {
        let tokens = enc.encode_ordinary(text);
        let ids: Vec<String> = tokens.iter().map(|id| id.to_string()).collect();
        println!("{}", ids.join(" "));
    } else {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let text = line?;
            let tokens = enc.encode_ordinary(&text);
            let ids: Vec<String> = tokens.iter().map(|id| id.to_string()).collect();
            println!("{}", ids.join(" "));
        }
    }

    Ok(())
}

fn cmd_decode(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: decode <vocab_json> [token_id ...]");
        std::process::exit(1);
    }

    let vocab_file = &args[0];
    let (vocab, merges) = load_tokenizer(vocab_file)?;
    let enc = FastBPEEncoder::new_with_merges(vocab, &merges)?;

    if args.len() > 1 {
        let ids: Result<Vec<u32>, _> = args[1..].iter().map(|s| s.parse::<u32>()).collect();
        let ids = ids?;
        println!("{}", enc.decode(&ids));
    } else {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line?;
            let ids: Result<Vec<u32>, _> =
                line.split_whitespace().map(|s| s.parse::<u32>()).collect();
            let ids = ids?;
            println!("{}", enc.decode(&ids));
        }
    }

    Ok(())
}

fn cmd_export(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 2 {
        eprintln!("Usage: export <vocab_json> <out_dir> [model_max_length]");
        std::process::exit(1);
    }

    let src = &args[0];
    let out_dir = &args[1];
    let max_len = args.get(2).and_then(|s| s.parse().ok());

    export_hf(src, out_dir, max_len)?;
    Ok(())
}

fn cmd_score(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 2 {
        eprintln!("Usage: score <vocab_json> <corpus_file> [label] [max_bytes]");
        std::process::exit(1);
    }

    let vocab_file = &args[0];
    let corpus_file = &args[1];
    let label = args.get(2).cloned().unwrap_or_else(|| corpus_file.clone());
    let max_bytes: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let (vocab, merges) = load_tokenizer(vocab_file)?;
    let enc = FastBPEEncoder::new_with_merges(vocab, &merges)?;

    let text = std::fs::read_to_string(corpus_file)?;
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

    // Warmup
    for l in lines.iter().take(500) {
        let _ = enc.encode_ordinary(l);
    }

    let mut best_mbs = f64::MIN;
    let mut total_tokens = 0usize;
    let mut total_words = 0usize;

    for _ in 0..3 {
        let start = Instant::now();
        let mut tokens = 0usize;
        let mut words = 0usize;
        for l in &lines {
            tokens += enc.encode_ordinary(l).len();
            words += l.split_whitespace().count();
        }
        let elapsed = start.elapsed().as_secs_f64();
        let mbs = (bytes as f64 / (1024.0 * 1024.0)) / elapsed;
        if mbs > best_mbs {
            best_mbs = mbs;
        }
        total_tokens = tokens;
        total_words = words;
    }

    let b_per_tok = if total_tokens > 0 {
        bytes as f64 / total_tokens as f64
    } else {
        0.0
    };
    let tok_per_word = if total_words > 0 {
        total_tokens as f64 / total_words as f64
    } else {
        0.0
    };

    println!(
        "{:<15} bytes={:<8} tokens={:<8} B/tok={:<6.2} tok/word={:<5.2} throughput={:<6.1} MB/s",
        label, bytes, total_tokens, b_per_tok, tok_per_word, best_mbs
    );

    Ok(())
}

fn cmd_repl(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let vocab_file = args
        .first()
        .map(String::as_str)
        .unwrap_or("output/tamil_tokenizer.json");

    println!("Loading vocabulary from {}...", vocab_file);
    let (vocab, merges) = load_tokenizer(vocab_file)?;
    let enc = FastBPEEncoder::new_with_merges(vocab, &merges)?;

    println!(
        "Loaded {} tokens. Type text to encode, or 'quit' to exit:\n",
        enc.vocab_size()
    );

    let stdin = io::stdin();
    loop {
        print!("> ");
        let _ = io::stdout().flush();

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() || input.eq_ignore_ascii_case("quit") {
            break;
        }

        let ids = enc.encode_ordinary(input);
        let decoded = enc.decode(&ids);
        println!("Tokens ({}): {:?}", ids.len(), ids);
        println!("Decoded: {}", decoded);
        println!();
    }

    Ok(())
}
