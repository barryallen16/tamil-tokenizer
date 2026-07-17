use tamil_tokenizer::{TamilTokenizer, OptimizedBPETrainer};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 4 {
        eprintln!("Usage: {} <corpus_file> <num_merges> <output_vocab>", args[0]);
        eprintln!("Example: {} ./data/tamil_corpus.txt 50000 ./output/tamil_tokenizer.json", args[0]);
        std::process::exit(1);
    }
    
    let corpus_file = &args[1];
    let num_merges: usize = match args[2].parse() {
        Ok(n) => n,
        Err(_) => { eprintln!("Error: '{}' is not a valid number of merges", args[2]); std::process::exit(1); }
    };
    let output_file = &args[3];
    
    println!("=== Training Tamil Tokenizer ===");
    println!("Corpus: {}", corpus_file);
    println!("Merges: {}", num_merges);
    println!("Output: {}\n", output_file);
    
    // Initialize with base vocabulary
    let tokenizer = TamilTokenizer::new();
    let _sorted_vocab = tokenizer.get_sorted_vocab(); 
    let vocab_ahash: ahash::AHashMap<String, u32> = tokenizer.vocab
        .into_iter()
        .collect();

    let mut trainer = OptimizedBPETrainer::new(vocab_ahash);

    
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
    
    println!("✓ Training complete!");
    println!("✓ Vocabulary saved to {}", output_file);
}
