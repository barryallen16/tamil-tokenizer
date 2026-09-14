#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;
    use tamil_tokenizer::{FastBPEEncoder, TamilPretokenizer, TamilTokenizer};

    fn make_encoder() -> FastBPEEncoder {
        let tok = TamilTokenizer::new();
        let vocab: FxHashMap<String, u32> = tok.get_sorted_vocab().into_iter().collect();
        FastBPEEncoder::new(vocab).expect("Failed to create test encoder")
    }

    // === TamilTokenizer (6 tests) ===

    #[test]
    fn test_vocab_size_range() {
        let tok = TamilTokenizer::new();
        assert!(tok.vocab_size() > 500);
        assert!(tok.vocab_size() < 600);
    }

    /// Losslessness rests on this: every one of the 256 bytes must have a
    /// single-char token, otherwise the encoder has nothing to fall back on
    /// and silently drops the byte.
    #[test]
    fn test_vocab_covers_all_256_bytes() {
        let tok = TamilTokenizer::new();
        for b in 0..=255u8 {
            let ch = tok.byte_encoder[&b];
            assert!(
                tok.vocab.contains_key(&ch.to_string()),
                "no single-char token for byte {}",
                b
            );
        }
    }

    #[test]
    fn test_sorted_vocab_no_duplicates() {
        let tok = TamilTokenizer::new();
        let sorted = tok.get_sorted_vocab();
        let mut ids: Vec<u32> = sorted.values().copied().collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), sorted.len());
    }

    #[test]
    fn test_sorted_vocab_ids_contiguous() {
        let tok = TamilTokenizer::new();
        let sorted = tok.get_sorted_vocab();
        let mut ids: Vec<u32> = sorted.values().copied().collect();
        ids.sort();
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, i as u32);
        }
    }

    #[test]
    fn test_byte_mappings_roundtrip() {
        let tok = TamilTokenizer::new();
        for b in 0..=255u8 {
            if let Some(&ch) = tok.byte_encoder.get(&b) {
                let back = tok.byte_decoder.get(&ch);
                assert!(back.is_some(), "No reverse for byte {}", b);
                assert_eq!(*back.unwrap(), b);
            }
        }
    }

    #[test]
    fn test_decode_token_space() {
        let tok = TamilTokenizer::new();
        if let Some(&space_char) = tok.byte_encoder.get(&b' ') {
            let decoded = tok.decode_token(&space_char.to_string());
            assert_eq!(decoded, vec![b' ']);
        }
    }

    #[test]
    fn test_save_and_load_vocab() {
        let tok = TamilTokenizer::new();
        let path = std::env::temp_dir().join("test_vocab.json");
        let path_str = path.to_str().unwrap();
        tok.save_vocab(path_str).expect("save failed");
        let loaded = TamilTokenizer::load_vocab_id_list(path_str).expect("load failed");
        assert_eq!(tok.vocab.len(), loaded.len());
        for (token, &id) in &tok.vocab {
            assert_eq!(loaded.get(token), Some(&id));
        }
        std::fs::remove_file(&path).ok();
    }

    // === Pretokenizer (5 tests) ===

    #[test]
    fn test_pretokenizer_empty() {
        let p = TamilPretokenizer::new();
        assert!(p.pretokenize("").is_empty());
    }

    #[test]
    fn test_pretokenizer_numbers() {
        let p = TamilPretokenizer::new();
        let c = p.pretokenize("12345");
        assert_eq!(c.concat(), "12345");
    }

    #[test]
    fn test_pretokenizer_punctuation() {
        let p = TamilPretokenizer::new();
        let c = p.pretokenize("!@#$%");
        assert!(!c.is_empty());
    }

    #[test]
    fn test_pretokenizer_whitespace() {
        let p = TamilPretokenizer::new();
        let c = p.pretokenize("   ");
        assert!(!c.is_empty());
        assert_eq!(c.concat(), "   ");
    }

    #[test]
    fn test_pretokenizer_numbers_and_punct() {
        let p = TamilPretokenizer::new();
        let c = p.pretokenize("123 + 456 = 789");
        let rejoined: String = c.concat();
        assert_eq!(rejoined, "123 + 456 = 789");
    }

    /// Regression: the apostrophe rule was missing its leading '|', so it was
    /// concatenated onto the special-token rule rather than being an alternative.
    /// That made contractions split, and stopped the special-token rule from ever
    /// matching on its own.
    #[test]
    fn test_pretokenizer_contraction_is_one_chunk() {
        let p = TamilPretokenizer::new();
        assert_eq!(p.pretokenize("it's"), vec!["it's"]);
        assert_eq!(p.pretokenize("don't"), vec!["don't"]);
    }

    #[test]
    fn test_pretokenizer_special_token_is_its_own_chunk() {
        let p = TamilPretokenizer::new();
        assert_eq!(p.pretokenize("<|endoftext|>"), vec!["<|endoftext|>"]);
        assert_eq!(
            p.pretokenize("<|endoftext|> hello"),
            vec!["<|endoftext|>", " hello"]
        );
    }

    /// The special-token pattern is [a-z_]+ with no digits, so reserved tokens
    /// like <|reserved_0|> cannot match it. They still round-trip, just as
    /// several tokens instead of one.
    #[test]
    fn test_pretokenizer_reserved_token_with_digits_roundtrips() {
        let p = TamilPretokenizer::new();
        let rejoined: String = p.pretokenize("<|reserved_0|>").concat();
        assert_eq!(rejoined, "<|reserved_0|>");
    }

    // === Encoder (12 tests) ===

    #[test]
    fn test_encoder_empty_string() {
        let enc = make_encoder();
        let ids = enc.encode_ordinary("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_encoder_bos_eos_wrapping() {
        let enc = make_encoder();
        let ids = enc.encode("123");
        assert!(
            ids.len() >= 3,
            "Expected BOS + tokens + EOS, got {}",
            ids.len()
        );
    }

    #[test]
    fn test_encoder_ordinary_no_bos_eos() {
        let enc = make_encoder();
        let ids = enc.encode_ordinary("123");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_encoder_decode_roundtrip() {
        let enc = make_encoder();
        let text = "123!";
        let ids = enc.encode_ordinary(text);
        let decoded = enc.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encoder_digit_encoding() {
        let enc = make_encoder();
        let ids = enc.encode_ordinary("123");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_encoder_special_token_lookup() {
        let enc = make_encoder();
        let bos = enc.get_special_token_id("<|startoftext|>");
        assert!(bos.is_some());
        let eos = enc.get_special_token_id("<|endoftext|>");
        assert!(eos.is_some());
    }

    #[test]
    fn test_encoder_is_special_token() {
        let enc = make_encoder();
        let bos = enc.get_special_token_id("<|startoftext|>").unwrap();
        assert!(enc.is_special_token(bos));
        assert!(!enc.is_special_token(0));
    }

    #[test]
    fn test_encoder_vocab_size() {
        let enc = make_encoder();
        assert!(enc.vocab_size() > 200);
    }

    #[test]
    fn test_encoder_special_token_count() {
        let enc = make_encoder();
        assert!(enc.special_token_count() >= 8);
    }

    #[test]
    fn test_encoder_reserved_token_count() {
        let enc = make_encoder();
        assert_eq!(enc.reserved_token_count(), 100);
    }

    #[test]
    fn test_encoder_decode_skip_special() {
        let enc = make_encoder();
        let ids = enc.encode("123");
        let clean = enc.decode_skip_special(&ids);
        assert_eq!(clean, "123");
    }

    #[test]
    fn test_encoder_long_digit_string() {
        let enc = make_encoder();
        let text = "1234567890";
        let ids = enc.encode(text);
        let decoded = enc.decode_skip_special(&ids);
        assert_eq!(decoded, text);
    }

    // === Losslessness (byte-level BPE) ===

    /// The bug these guard: the vocab stores GPT-2 style byte-mapped chars, but
    /// the encoder used to feed raw text straight to the automaton. Anything the
    /// pretokenizer split off into a chunk with no vocab match was dropped with
    /// no error, so encode/decode silently lost characters.
    fn assert_roundtrip(text: &str) {
        let enc = make_encoder();
        let ids = enc.encode_ordinary(text);
        assert_eq!(enc.decode(&ids), text, "roundtrip failed for {:?}", text);
    }

    #[test]
    fn test_roundtrip_tamil_with_spaces() {
        assert_roundtrip("வணக்கம் இது தமிழ்");
    }

    #[test]
    fn test_roundtrip_english_letters() {
        // Base vocab deliberately ships no learned merges, so each ASCII letter
        // must still survive via the single-byte fallback.
        assert_roundtrip("Hello world");
    }

    #[test]
    fn test_roundtrip_control_characters() {
        assert_roundtrip("a\nb");
        assert_roundtrip("a\tb\rc");
        assert_roundtrip("line1\nline2\n\nline4");
    }

    #[test]
    fn test_roundtrip_emoji() {
        assert_roundtrip("😀");
        assert_roundtrip("hi 😀🎉 there");
    }

    #[test]
    fn test_roundtrip_mixed_script() {
        assert_roundtrip("வணக்கம் Hello 123 !@# 😀 தமிழ்\nend");
    }

    #[test]
    fn test_roundtrip_leading_and_trailing_space() {
        assert_roundtrip("  padded  ");
    }

    /// Tamil clusters are stored as raw text. The encoder used to byte-map input
    /// before matching, which made every Tamil token unreachable.
    #[test]
    fn test_tamil_cluster_is_one_token() {
        let enc = make_encoder();
        assert_eq!(enc.encode_ordinary("கா").len(), 1);
        assert_eq!(enc.encode_ordinary("ஃ").len(), 1);
    }

    #[test]
    fn test_tamil_compresses_beyond_bytes() {
        let enc = make_encoder();
        let text = "வணக்கம் இது தமிழ்";
        let ratio = text.len() as f64 / enc.encode_ordinary(text).len() as f64;
        assert!(
            ratio > 2.0,
            "Tamil bytes/token {:.2} should exceed 2.0",
            ratio
        );
    }

    /// Byte tokens use stand-in chars U+0100..U+01A1. Real text containing those
    /// chars must not match the byte tokens, or decode emits raw bytes instead.
    #[test]
    fn test_roundtrip_latin_extended() {
        assert_roundtrip("Ş ğ ő Ā");
        assert_roundtrip("zażółć gęślą jaźń");
        assert_roundtrip("Ġ Ċ ĉ");
    }

    /// Rank order decides, not longest match: ("a","b") is the only merge, so
    /// "ab" leaves the base set and "abc" segments as a|bc with nothing left
    /// to merge. Longest-match would have emitted ab|c instead.
    #[test]
    fn test_merge_rank_beats_longest_match() {
        use tamil_tokenizer::FastBPEEncoder;
        let vocab: FxHashMap<String, u32> = ["a", "b", "c", "ab", "bc"]
            .into_iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        let enc = FastBPEEncoder::new_with_merges(vocab, &[("a".to_string(), "b".to_string())])
            .expect("Failed to create test encoder");
        assert_eq!(enc.encode_ordinary("abc"), vec![0, 4]);
    }

    #[test]
    fn test_chained_merges_collapse() {
        use tamil_tokenizer::FastBPEEncoder;
        let vocab: FxHashMap<String, u32> = ["a", "b", "c", "ab", "abc"]
            .into_iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let enc =
            FastBPEEncoder::new_with_merges(vocab, &merges).expect("Failed to create test encoder");
        assert_eq!(enc.encode_ordinary("abc"), vec![4]);
    }

    #[test]
    fn test_encode_never_silently_drops_text() {
        // Total decoded bytes must always equal input bytes, whatever the input.
        let enc = make_encoder();
        for text in ["a", " ", "\n", "😀", "ஃ", "Hello", "வணக்கம்", "x y\nz"] {
            let ids = enc.encode_ordinary(text);
            assert_eq!(
                enc.decode(&ids).len(),
                text.len(),
                "byte count changed for {:?}",
                text
            );
        }
    }

    /// A vocab with an ID past the end used to index `id_to_token` directly and
    /// panic with an out-of-bounds index. It must be rejected as bad data
    /// instead, because a malformed tokenizer file is untrusted input.
    #[test]
    fn test_malformed_vocab_is_rejected_not_panicked() {
        let mut sparse: FxHashMap<String, u32> = FxHashMap::default();
        sparse.insert("a".to_string(), 0);
        sparse.insert("b".to_string(), u32::MAX);
        assert!(
            FastBPEEncoder::new(sparse).is_err(),
            "sparse ids must be rejected"
        );

        // Dense prefix but a hole before the end: would silently decode to "".
        let mut gap: FxHashMap<String, u32> = FxHashMap::default();
        gap.insert("a".to_string(), 0);
        gap.insert("b".to_string(), 7);
        assert!(
            FastBPEEncoder::new(gap).is_err(),
            "gapped ids must be rejected"
        );
    }
}
