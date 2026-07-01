use tamil_tokenizer::{TamilTokenizer, TamilPretokenizer};
use std::fs;

fn main() {
    // Initialize tokenizer
    let tokenizer = TamilTokenizer::new();
    tokenizer.print_stats();

    if let Err(e) = fs::create_dir_all("./output") {
        eprintln!("Failed to create output directory: {}", e);
        return;
    }

    // Save vocabulary file
    match tokenizer.save_vocab("./output/tamil_vocab.json") {
        Ok(_) => println!("\n✓ Vocabulary saved to ./output/tamil_vocab.json"),
        Err(e) => eprintln!("Error saving vocab: {}", e),
    }

    // Tamil Pre-tokenization test
    println!("\n=== Tamil Pretokenization Test ===");
    let pretok = TamilPretokenizer::new();
    let sample_text = "வணக்கம்! இது தமிழ் உரை. ABC தமிழ் 123 யாத்திரை 𑿱";
    pretok.print_example(sample_text);

    // Pretokenize a corpus input file
    // fs::write("./data/tamil_corpus.txt", "வணக்கம்! இது தமிழ் உரை.\nநன்றி.").unwrap();
    match pretok.pretokenize_file("./data/tamil_corpus.txt", "./output/pretokenized.txt") {
        Ok(_) => println!("\n✓ Pretokenized corpus saved to ./output/pretokenized.txt"),
        Err(e) => eprintln!("Pretokenization failed: {}", e),
    }
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