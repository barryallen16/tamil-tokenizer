use std::env;
use tamil_tokenizer::{load_tokenizer, FastBPEEncoder};

/// Read one line of text per stdin line and print `{"line": [...ids]}` so a
/// verification script can compare against another tokenizer.
fn main() {
    let path = env::args()
        .nth(1)
        .expect("Usage: encode_lines <tokenizer.json>");
    let (vocab, merges) = load_tokenizer(&path).expect("load tokenizer");
    let enc = FastBPEEncoder::new_with_merges(vocab, &merges).expect("build encoder");

    let stdin = std::io::stdin();
    let mut map = serde_json::Map::new();
    for line in stdin.lines() {
        let line = line.expect("read line");
        if line.is_empty() {
            continue;
        }
        map.insert(line.clone(), serde_json::json!(enc.encode_ordinary(&line)));
    }
    println!("{}", serde_json::to_string(&map).expect("serialize"));
}
