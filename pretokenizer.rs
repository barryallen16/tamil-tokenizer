use regex::Regex;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};

pub struct TamilPretokenizer {
    pattern: Regex,
}

impl Default for TamilPretokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TamilPretokenizer {
    pub fn new() -> Self {
        // Every alternative after the first must be introduced by '|'. Omitting it
        // concatenates two rules instead of separating them.
        let pattern_str = concat!(
            r"<\|[a-z_]+\|>",
            r"| ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+", // words with apostrophes
            r"|_(?:\p{Ll}\p{M}*)+",                          // words starting with underscore
            r"| ?(?:\p{Lu}\p{M}*)+",                         // uppercase sequences
            r"| ?(?:\p{Ll}\p{M}*)+",                         // lowercase or mixed
            r"| ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",           // Tamil/Indic letters
            r"|(?:\p{N}\p{M}*)+",                            // numbers
            r"| ?(?:[\p{P}\p{S}]\p{M}*)+",                   // punctuation and symbols
            r"|[^\S\r\n]*[\n\r]+|[^\S\r\n]+",                // whitespace/newlines
            r"|(?:[\p{Z}\p{C}]\p{M}*)+",                     // separators/controls
            r"|\p{M}+"                                       // stray marks
        );

        let pattern = Regex::new(pattern_str).unwrap();

        Self { pattern }
    }

    pub fn pretokenize(&self, text: &str) -> Vec<String> {
        self.pattern
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }

    pub fn pretokenize_file(&self, input_path: &str, output_path: &str) -> io::Result<()> {
        let input_file = fs::File::open(input_path)?;
        let reader = BufReader::new(input_file);
        let output_file = fs::File::create(output_path)?;
        let mut writer = io::BufWriter::new(output_file);

        for line in reader.lines() {
            let line = line?;
            let chunks = self.pretokenize(&line);
            for chunk in chunks {
                writeln!(writer, "{}", chunk)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub fn print_example(&self, text: &str) {
        let chunks = self.pretokenize(text);
        println!("Pretokenized {} chunks:", chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("  {}: {}", i, chunk);
        }
    }
}
