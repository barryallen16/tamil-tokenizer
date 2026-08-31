// use tamil_tokenizer::{TamilTokenizer, TamilPretokenizer};
// use std::fs;

// fn main() {
//     // Initialize tokenizer
//     let tokenizer = TamilTokenizer::new();
//     tokenizer.print_stats();

//     if let Err(e) = fs::create_dir_all("./output") {
//         eprintln!("Failed to create output directory: {}", e);
//         return;
//     }

//     // Save vocabulary file
//     match tokenizer.save_vocab("./output/tamil_vocab.json") {
//         Ok(_) => println!("Vocabulary saved to ./output/tamil_vocab.json"),
//         Err(e) => eprintln!("Error saving vocab: {}", e),
//     }

//     // Tamil Pre-tokenization test
//     println!("\n=== Tamil Pretokenization Test ===");
//     let pretok = TamilPretokenizer::new();
//     let sample_text = "வணக்கம்! இது தமிழ் உரை. ABC தமிழ் 123 யாத்திரை 𑿱";
//     pretok.print_example(sample_text);

//     // Pretokenize a corpus input file
//     // fs::write("./data/tamil_corpus.txt", "வணக்கம்! இது தமிழ் உரை.\nநன்றி.").unwrap();
//     match pretok.pretokenize_file("./data/tamil_corpus.txt", "./output/pretokenized.txt") {
//         Ok(_) => println!("Pretokenized corpus saved to ./output/pretokenized.txt"),
//         Err(e) => eprintln!("Pretokenization failed: {}", e),
//     }
// }

// use tamil_tokenizer::{TamilTokenizer};
// // use std::fs;

// fn main() {
//     let tokenizer = TamilTokenizer::new();
//     tokenizer.print_stats();
//     tokenizer.save_vocab("./output/tamil_vocab_cluster.json").unwrap();
// }
use tamil_tokenizer::FastBPEEncoder;
// use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use serde_json;
use rustc_hash::FxHashMap;

fn main() {
    println!("=== Tamil Fast Encoder/Decoder ===\n");
    
    // Load your trained vocabulary
    println!("Loading vocabulary from output/tamil_tokenizer.json...");
    let data = match fs::read_to_string("output/tamil_tokenizer.json") {
        Ok(d) => d,
        Err(e) => { eprintln!("Failed to read vocab file: {}", e); return; }
    };
    
    let vocab: FxHashMap<String, u32> = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to parse vocab JSON: {}", e); return; }
    };
    
    println!("✓ Loaded {} tokens\n", vocab.len());
    
    let encoder = match FastBPEEncoder::new(vocab) {
        Ok(e) => e,
        Err(e) => { eprintln!("Failed to create encoder: {}", e); return; }
    };
    
    // Interactive mode
    println!("Enter your text to encode (or 'quit' to exit):\n");
    
    loop {
        print!("> ");
        let _ = io::stdout().flush();
        
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() { break; }
        let input = input.trim();
        
        if input.is_empty() || input.eq_ignore_ascii_case("quit") {
            break;
        }
        
        // Encode
        let token_ids = encoder.encode(input);
        println!("\n✓ Encoded to {} tokens:", token_ids.len());
        println!("  {:?}", token_ids);
        
        // Decode
        let decoded = encoder.decode(&token_ids);
        println!("\n✓ Decoded:");
        println!("  {}", decoded);
        
        // Verify
        println!("\n✓ Match: {}\n", if input == decoded { "✅" } else { "❌" });
        println!("{}", "=".repeat(60));
        println!();
    }
    
    println!("Goodbye!");
}





// use tamil_tokenizer::TamilPretokenizer;
// use std::fs::{File, create_dir_all};
// use std::io::{BufRead, BufReader, BufWriter, Write};

// fn main() {
//     // Initialize the pretokenizer
//     let pretokenizer = TamilPretokenizer::new();

//     // Input and output paths
//     let input_path = "./data/tamil_corpus.txt";
//     let output_path = "./output/pretokenized.txt";

//     // Ensure output directory exists
//     create_dir_all("./output").expect("Failed to create output directory");

//     // Open input and output files
//     let input_file = File::open(input_path).expect("Failed to open corpus file");
//     let reader = BufReader::new(input_file);
//     let output_file = File::create(output_path).expect("Failed to create output file");
//     let mut writer = BufWriter::new(output_file);

//     // Process each line from corpus and write chunks to file
//     for line in reader.lines() {
//         let line = line.unwrap();
//         if line.trim().is_empty() {
//             continue;
//         }

//         let chunks = pretokenizer.pretokenize(&line);
//         for chunk in &chunks {
//             writeln!(writer, "{}", chunk).unwrap();
//         }
//     }

//     writer.flush().unwrap();
// }
