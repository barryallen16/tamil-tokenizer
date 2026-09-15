use std::env;
use tamil_tokenizer::{OptimizedBPETrainer, TamilTokenizer};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!(
            "Usage: {} <corpus_file> <num_merges> <output_vocab> [min_frequency]",
            args[0]
        );
        eprintln!(
            "Example: {} ./data/tamil_corpus.txt 50000 ./output/tamil_tokenizer.json",
            args[0]
        );
        eprintln!("min_frequency defaults to 1 (learn every merge); pass 2+ to skip rare pairs.");
        std::process::exit(1);
    }

    let corpus_file = &args[1];
    let num_merges: usize = match args[2].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error: '{}' is not a valid number of merges", args[2]);
            std::process::exit(1);
        }
    };
    let output_file = &args[3];
    let min_frequency: u32 = match args.get(4) {
        Some(v) => match v.parse::<u32>() {
            Ok(n) => n.max(1),
            Err(_) => {
                eprintln!("Error: '{}' is not a valid min_frequency", v);
                std::process::exit(1);
            }
        },
        None => 1,
    };

    println!("=== Training Tamil Tokenizer ===");
    println!("Corpus: {}", corpus_file);
    println!("Merges: {}", num_merges);
    println!("Min frequency: {}", min_frequency);
    println!("Output: {}\n", output_file);

    // Initialize with base vocabulary
    let tokenizer = TamilTokenizer::new();
    let vocab_ahash: ahash::AHashMap<String, u32> = tokenizer.vocab.into_iter().collect();

    let mut trainer = OptimizedBPETrainer::new(vocab_ahash);
    trainer.set_min_frequency(min_frequency);

    // Train
    println!("Loading corpus...");
    if let Err(e) = trainer.initialize_from_corpus(corpus_file) {
        eprintln!("Failed to load corpus: {}", e);
        std::process::exit(1);
    }

    println!("Training BPE...");
    trainer.train(num_merges);

    println!("\nSaving vocabulary...");
    if let Err(e) = trainer.save_tokenizer_with_reserved_tokens(output_file) {
        eprintln!("Failed to save tokenizer: {}", e);
        std::process::exit(1);
    }

    println!("Training complete!");
    println!("Vocabulary saved to {}", output_file);
}
