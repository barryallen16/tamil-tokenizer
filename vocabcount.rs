//! Prints the base-vocabulary breakdown from `TamilTokenizer::print_stats` and
//! checks it against the size the vocab actually has.
//!
//! Run: cargo run --release --bin vocabcount

use tamil_tokenizer::TamilTokenizer;

fn main() {
    let tok = TamilTokenizer::new();
    tok.print_stats();

    // The buckets printed above are hand-maintained and have drifted from the
    // builder before. This asserts the one invariant that matters: they add up.
    const SUM: usize = 256 + 12 + 23 + 276 + 4 + 15 + 528 + 10 + 11;
    let actual = tok.vocab_size();
    assert_eq!(
        SUM, actual,
        "vocab breakdown no longer sums to the real vocab size"
    );
    println!("\ncheck: breakdown sum == vocab_size ({actual}) OK");
}
