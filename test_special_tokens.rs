use tamil_tokenizer::{load_tokenizer, FastBPEEncoder};

fn main() {
    let (vocab, merges) = match load_tokenizer("./output/tamil_tokenizer.json") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load vocab: {}", e);
            std::process::exit(1);
        }
    };
    let encoder = match FastBPEEncoder::new_with_merges(vocab, &merges) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create encoder: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== Special Token Tests ===\n");

    // Test 1: Auto BOS/EOS
    println!("1. Auto BOS/EOS:");
    let tokens = encoder.encode(" வணக்கம். நான் ஒரு தமிழ் tokenizer");
    let tokens1 = encoder.encode_ordinary(" வணக்கம். நான் ஒரு தமிழ் tokenizer");
    println!("   Tokens with specials: {:?}", tokens);
    println!("   Tokens without specials: {:?}", tokens1);
    println!("   Decoded (with special): {}", encoder.decode(&tokens));
    println!(
        "   Decoded (clean): {}\n",
        encoder.decode_skip_special(&tokens)
    );

    // Test 2: Question-Answer
    println!("2. Question-Answer:");
    let tokens = encoder.encode("யார் நீ? நான் ஒரு தமிழ் tokenizer");
    let tokens1 = encoder.encode_ordinary("யார் நீ? நான் ஒரு தமிழ் tokenizer");
    println!("   Tokens with specials: {:?}", tokens);
    println!("   Tokens without specials: {:?}", tokens1);
    println!("   Decoded (with special): {}", encoder.decode(&tokens));
    println!("   Clean: {}\n", encoder.decode_skip_special(&tokens));

    // Test 3: Language mixing
    println!("3. Language Tags:");
    let tokens = encoder.encode("வணக்கம் Hello world");
    let tokens1 = encoder.encode_ordinary("வணக்கம் Hello world");
    println!("   Tokens: {:?}", tokens);
    println!("   Tokens without specials: {:?}", tokens1);
    println!("   Decoded (with special): {}", encoder.decode(&tokens));
    println!("   Clean: {}\n", encoder.decode_skip_special(&tokens));

    // Test 4: Vocab stats
    println!("4. Tokenizer Stats:");
    println!("   Total vocab: {}", encoder.vocab_size());
    println!("   Decoded (with special): {}", encoder.decode(&tokens));
    println!("   Special tokens: {}", encoder.special_token_count());
}
