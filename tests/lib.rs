#[cfg(test)]
mod tests {
    use tamil_tokenizer::{TamilTokenizer, TamilPretokenizer, FastBPEEncoder};
    use rustc_hash::FxHashMap;

    fn make_encoder() -> FastBPEEncoder {
        let tok = TamilTokenizer::new();
        let vocab: FxHashMap<String, u32> = tok.get_sorted_vocab().into_iter().collect();
        FastBPEEncoder::new(vocab).expect("Failed to create test encoder")
    }

    // === TamilTokenizer (6 tests) ===

    #[test]
    fn test_vocab_size_range() {
        let tok = TamilTokenizer::new();
        assert!(tok.vocab_size() > 200);
        assert!(tok.vocab_size() < 400);
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
        assert!(ids.len() >= 3, "Expected BOS + tokens + EOS, got {}", ids.len());
    }

    #[test]
    fn test_encoder_ordinary_no_bos_eos() {
        let enc = make_encoder();
        let ids = enc.encode_ordinary("123");
        assert!(ids.len() >= 1);
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
}
