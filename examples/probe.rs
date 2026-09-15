fn main() {
    for _ in 0..3 {
        let t = tamil_tokenizer::TamilTokenizer::new();
        println!(
            "vocab.len()={} vocab_size()={}",
            t.vocab.len(),
            t.vocab_size()
        );
    }
}
