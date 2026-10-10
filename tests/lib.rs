#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;
    use tamil_tokenizer::{FastBPEEncoder, TamilPretokenizer, TamilTokenizer};

    fn make_encoder() -> FastBPEEncoder {
        let tok = TamilTokenizer::new();
        let vocab: FxHashMap<String, u32> = tok.vocab.clone().into_iter().collect();
        FastBPEEncoder::new(vocab).expect("Failed to create test encoder")
    }

    fn base_vocab() -> ahash::AHashMap<String, u32> {
        TamilTokenizer::new().vocab.into_iter().collect()
    }

    // === TamilTokenizer (6 tests) ===

    #[test]
    fn test_vocab_size_range() {
        let tok = TamilTokenizer::new();
        // 256 byte + 12 vowels + 23 consonants + 276 consonant-vowel + 4 special
        // + 15 ligatures + 529 conjunct stems (23 x 23) + 21 digits and
        // symbols, minus க்ஷ which is both a ligature and a conjunct stem.
        assert_eq!(tok.vocab_size(), 1135);
    }

    /// Conjunct stems are the fix that actually moved compression; Grantha
    /// consonants alone barely did. Tamil joins consonants constantly
    /// (க்ஷ, ஸ்ரீ, க்ட, ண்ட) and the base set previously had to spend 3-4
    /// tokens on each one.
    #[test]
    fn test_conjunct_stems_are_single_tokens() {
        let enc = make_encoder();
        for stem in ["க்ஷ", "ஸ்ர", "க்ட", "ண்ட", "ஜ்ஜ", "ஷ்ட", "ஸ்த", "ஷ்ம", "ஶ்ந"]
        {
            assert_eq!(
                enc.encode_ordinary(stem).len(),
                1,
                "conjunct {stem:?} should be one token"
            );
        }
        assert_eq!(enc.encode_ordinary("க்ஷ").len(), 1);
    }

    #[test]
    fn test_conjunct_words_roundtrip() {
        let enc = make_encoder();
        for word in ["விஷயம்", "பாஷணம்", "கண்டனம்", "தமிழ்நாடு", "ஸ்ரீராம்"]
        {
            assert_eq!(enc.decode(&enc.encode_ordinary(word)), word);
        }
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

    /// The encoder indexes `id_to_token` by ID, so the base vocab's IDs must be
    /// unique and dense from 0. `new()` assigns them sequentially.
    #[test]
    fn test_base_vocab_ids_unique_and_contiguous() {
        let tok = TamilTokenizer::new();
        let mut ids: Vec<u32> = tok.vocab.values().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), tok.vocab.len(), "duplicate ids in base vocab");
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, i as u32, "base vocab ids are not contiguous");
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
        // Grantha consonants and vowel clusters
        assert_eq!(enc.encode_ordinary("ஜா").len(), 1);
        assert_eq!(enc.encode_ordinary("ஜி").len(), 1);
        assert_eq!(enc.encode_ordinary("ஷா").len(), 1);
        assert_eq!(enc.encode_ordinary("ஸு").len(), 1);
        assert_eq!(enc.encode_ordinary("ஹோ").len(), 1);
        assert_eq!(enc.encode_ordinary("ஶ்").len(), 1);
        // Ligatures
        assert_eq!(enc.encode_ordinary("ஸ்ரீ").len(), 1);
        assert_eq!(enc.encode_ordinary("ஶ்ரீ").len(), 1);
        assert_eq!(enc.encode_ordinary("க்ஷ").len(), 1);
        assert_eq!(enc.encode_ordinary("க்ஷா").len(), 1);
    }

    #[test]
    fn test_grantha_words_roundtrip() {
        let enc = make_encoder();
        for word in ["ரோஜா", "ஹலோ", "விஷயம்", "ஸ்டேஷன்", "பேஸ்புக்", "ஸ்ரீராம்", "க்ஷேமம்"]
        {
            let ids = enc.encode_ordinary(word);
            let decoded = enc.decode(&ids);
            assert_eq!(decoded, word);
        }
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
    /// The byte table must stay identical to GPT-2's, byte for byte. Hugging
    /// Face's `ByteLevel` pretokenizer and decoder hardcode it, so any drift
    /// here shows up only after export as silently wrong IDs in Python.
    #[test]
    fn test_byte_table_matches_gpt2() {
        let (encoder, decoder) = TamilTokenizer::create_byte_mappings();
        assert_eq!(encoder.len(), 256);
        assert_eq!(decoder.len(), 256);
        assert_eq!(decoder.len(), encoder.len(), "table must be a bijection");

        // GPT-2's printable sets; everything else goes to U+0100 upward in
        // ascending byte order.
        let printable =
            |b: u8| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);

        let mut next = 0u32;
        for b in 0..=255u8 {
            let expected = if printable(b) {
                b as char
            } else {
                let c = std::char::from_u32(256 + next).unwrap();
                next += 1;
                c
            };
            assert_eq!(encoder[&b], expected, "byte {b} maps wrong");
            assert_eq!(decoder[&expected], b, "byte {b} does not round-trip");
        }

        // Spot checks against known GPT-2 values.
        assert_eq!(encoder[&b' '], '\u{0120}');
        assert_eq!(encoder[&b'\n'], '\u{010A}');
        assert_eq!(encoder[&b'A'], 'A');
        assert_eq!(encoder[&0xC3], '\u{00C3}');
    }

    /// Latin-1 text must not be captured by a single-byte token. Under GPT-2's
    /// table `Ã` is both the byte token for 0xC3 and a real character, so a
    /// stand-in test that only looked at the codepoint would match raw text to
    /// a one-byte token and decode it to a different byte.
    #[test]
    fn test_latin1_text_is_not_a_byte_token() {
        for text in ["Ã©", "Ñ€", "üÿ", "Ð¿Ñ€Ð¸Ð²ÐµÑ‚"] {
            assert_roundtrip(text);
        }
    }

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

    // === OptimizedBPETrainer ===

    const CORPUS: &str = "./data/tamil_corpus.txt";

    fn trained(merges: usize, min_frequency: u32) -> tamil_tokenizer::OptimizedBPETrainer {
        let base: ahash::AHashMap<String, u32> = TamilTokenizer::new().vocab.into_iter().collect();
        let mut t = tamil_tokenizer::OptimizedBPETrainer::new(base);
        t.set_min_frequency(min_frequency);
        t.initialize_from_corpus(CORPUS)
            .expect("sample corpus should be readable");
        t.train(merges);
        t
    }

    #[test]
    fn test_trainer_learns_the_requested_merges() {
        let base_len = TamilTokenizer::new().vocab.len();
        let t = trained(50, 1);
        let merges = t.merges();
        assert_eq!(merges.len(), 50, "should record exactly 50 merges");

        // The final vocab is larger than the base vocab by more than the merge
        // count: `convert_to_ids` also mints IDs for corpus clusters that are
        // not in the base vocab, so an exact delta would be brittle.
        let final_vocab = t.get_final_vocab();
        assert!(
            final_vocab.len() > base_len + 50,
            "vocab should grow past base + merges"
        );

        // The real invariant: every recorded merge produced its token.
        for (a, b) in merges {
            let merged = format!("{a}{b}");
            assert!(
                final_vocab.contains_key(&merged),
                "merge {a:?}+{b:?} is missing token {merged:?}"
            );
        }
    }

    /// The trainer breaks frequency ties by packed pair key, so the same corpus
    /// must produce byte-identical merges every run. Reproducible published
    /// models depend on this.
    #[test]
    fn test_trainer_is_deterministic() {
        assert_eq!(trained(60, 1).merges(), trained(60, 1).merges());
    }

    #[test]
    fn test_min_frequency_skips_rare_pairs() {
        let loose = trained(300, 1);
        let strict = trained(300, 5);
        assert!(
            strict.get_final_vocab().len() < loose.get_final_vocab().len(),
            "a frequency floor must shrink the vocab"
        );
    }

    /// End-to-end: a trained artifact saved and reloaded must encode and decode
    /// its own corpus byte-exactly. This is the path the CLI and every benchmark
    /// depend on, and nothing else tested it.
    #[test]
    fn test_trained_artifact_round_trips() {
        let t = trained(200, 1);
        let path = std::env::temp_dir().join("tamil_tokenizer_trainer_test.json");
        let path = path.to_str().unwrap();
        t.save_tokenizer_with_reserved_tokens(path).unwrap();

        let (vocab, merges) =
            tamil_tokenizer::load_tokenizer(path).expect("trained file should load");
        assert!(
            !merges.is_empty(),
            "merges must survive the save/load round-trip"
        );

        let enc = FastBPEEncoder::new_with_merges(vocab, &merges).expect("encoder should build");
        let corpus = std::fs::read_to_string(CORPUS).unwrap();

        let mut checked = 0;
        for line in corpus.lines().filter(|l| !l.trim().is_empty()).take(200) {
            let ids = enc.encode_ordinary(line);
            assert_eq!(enc.decode(&ids), line, "round-trip changed {:?}", line);
            checked += 1;
        }
        assert!(
            checked > 50,
            "expected a meaningful sample, got {checked} lines"
        );

        let _ = std::fs::remove_file(path);
    }

    // === Trainer correctness (2 tests) ===

    /// Segment a chunk into base symbols exactly as the trainer does:
    /// longest-first over 4/3/2-character clusters, then a single character.
    fn segment(clusters: &ahash::AHashMap<String, u32>, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            for len in (2..=4).rev() {
                if !matched && i + len <= chars.len() {
                    let cluster: String = chars[i..i + len].iter().collect();
                    if clusters.contains_key(&cluster) {
                        out.push(cluster);
                        i += len;
                        matched = true;
                    }
                }
            }
            if !matched {
                out.push(chars[i].to_string());
                i += 1;
            }
        }
        out
    }

    #[inline]
    fn pack_pair(a: u32, b: u32) -> u64 {
        ((a as u64) << 32) | (b as u64)
    }

    fn train_once(
        base: ahash::AHashMap<String, u32>,
        path: &str,
        merges: usize,
    ) -> (Vec<(String, String)>, ahash::AHashMap<String, u32>) {
        let mut trainer = tamil_tokenizer::OptimizedBPETrainer::new(base);
        trainer
            .initialize_from_corpus(path)
            .expect("corpus should load");
        trainer.train(merges);
        (trainer.merges().to_vec(), trainer.get_final_vocab())
    }

    /// Reference greedy BPE, recomputed from scratch at every step.
    ///
    /// The real trainer keeps pair counts incrementally and picks the maximum
    /// from a heap. A bug in that bookkeeping is invisible from the outside:
    /// training still finishes, the model still round-trips, and only the
    /// compression silently suffers. The failure this exists to catch is the
    /// lazy heap dropping a stale entry instead of re-pushing it with its
    /// corrected count, which removes frequent pairs from consideration for
    /// good and cost 25% of bytes/token at 30K merges when it was live.
    ///
    /// Ties break on the packed key, smallest first, matching `HeapEntry::cmp`.
    /// Ids are assigned the way `OptimizedBPETrainer` assigns them: base
    /// vocabulary in lexicographic order, then unknown symbols minted while
    /// walking the unique words in sorted order, then one id per merge.
    fn brute_force_merges(
        base: &ahash::AHashMap<String, u32>,
        corpus: &str,
        n_merges: usize,
    ) -> Vec<(String, String)> {
        let pretok = TamilPretokenizer::new();
        let mut freq: std::collections::HashMap<Vec<String>, u64> =
            std::collections::HashMap::new();
        for line in corpus.lines() {
            for chunk in pretok.pretokenize(line) {
                *freq.entry(segment(base, chunk)).or_insert(0) += 1;
            }
        }

        let mut sorted: Vec<&String> = base.keys().collect();
        sorted.sort();
        let mut id_of: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut name_of: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        for (i, token) in sorted.iter().enumerate() {
            id_of.insert((*token).clone(), i as u32);
            name_of.insert(i as u32, (*token).clone());
        }
        let mut next_id = sorted.len() as u32;

        let mut entries: Vec<(Vec<String>, u64)> = freq.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut words: Vec<(Vec<u32>, u64)> = Vec::with_capacity(entries.len());
        for (tokens, f) in entries {
            let mut ids = Vec::with_capacity(tokens.len());
            for token in tokens {
                let id = match id_of.get(&token) {
                    Some(&id) => id,
                    None => {
                        let id = next_id;
                        next_id += 1;
                        id_of.insert(token.clone(), id);
                        name_of.insert(id, token);
                        id
                    }
                };
                ids.push(id);
            }
            words.push((ids, f));
        }

        let mut merges = Vec::new();
        for _ in 0..n_merges {
            let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
            for (tokens, f) in &words {
                for pair in tokens.windows(2) {
                    *counts.entry(pack_pair(pair[0], pair[1])).or_insert(0) += *f;
                }
            }

            let mut best: Option<(u64, u64)> = None;
            for (&key, &count) in &counts {
                match best {
                    None => best = Some((key, count)),
                    Some((best_key, best_count)) => {
                        if count > best_count || (count == best_count && key < best_key) {
                            best = Some((key, count));
                        }
                    }
                }
            }
            let Some((best_key, _)) = best else { break };

            let a = (best_key >> 32) as u32;
            let b = (best_key & 0xFFFF_FFFF) as u32;
            merges.push((name_of[&a].clone(), name_of[&b].clone()));
            let merged = format!("{}{}", name_of[&a], name_of[&b]);
            let new_id = next_id;
            next_id += 1;
            name_of.insert(new_id, merged);

            for (tokens, _) in words.iter_mut() {
                let mut out = Vec::with_capacity(tokens.len());
                let mut i = 0;
                while i < tokens.len() {
                    if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
                        out.push(new_id);
                        i += 2;
                    } else {
                        out.push(tokens[i]);
                        i += 1;
                    }
                }
                *tokens = out;
            }
        }
        merges
    }

    /// Corpus with conjunct-plus-matra words (`மற்றும்`, `தமிழ்நாட்டில்`). Those
    /// leave a vowel sign standing alone, which the base vocabulary has no token
    /// for, so the trainer mints ids for them mid-training.
    const REPRO_CORPUS: &str = "\
மற்றும் ஒரு நாடு தமிழ்
தமிழ்நாடு மற்றும் ஒரு மொழி
பற்றி மற்றும் இருந்து ஒரு
தமிழ்நாட்டில் மற்றும் பற்றிய
இருந்து தமிழ்நாட்டில் மற்றும்
ஒரு பற்றி மற்றும் இருந்து
";

    #[test]
    fn trainer_matches_brute_force_greedy_bpe() {
        // The tracked sample corpus, not a synthetic one. A stale heap entry
        // only has to be reconciled when a pair loses *part* of its count, which
        // needs that pair to occur in several distinct words. Twelve repeated
        // lines drop counts straight to zero instead (measured: 1 stale entry
        // over 150 merges), and the branch this test exists to cover is never
        // reached. The sample corpus reaches it ~124 times.
        //
        // Several merge counts, not just one. The heap defect shows up at
        // different densities: early merges are sparse and rarely go stale, late
        // ones are dense and go stale constantly. A single count could sit in a
        // range where the bug is invisible. 50 is sparse, 150 is where the
        // defect was originally found, 400 is dense enough that most pairs have
        // been touched by several merges.
        const MERGE_COUNTS: [usize; 3] = [50, 150, 400];
        let corpus = std::fs::read_to_string(CORPUS).expect("the sample corpus is tracked in git");

        for merges in MERGE_COUNTS {
            let expected = brute_force_merges(&base_vocab(), &corpus, merges);
            let (actual, _) = train_once(base_vocab(), CORPUS, merges);
            assert_eq!(
                actual, expected,
                "incremental trainer diverged from greedy BPE at {merges} merges"
            );
        }
    }

    /// Two runs over the same corpus must produce the same model.
    ///
    /// Ids are minted while walking the unique-word map, and the merge tie-break
    /// compares packed ids, so iterating a hash map in hash order makes the
    /// model depend on the seed ahash picked for that process.
    #[test]
    fn training_is_reproducible() {
        let path = std::env::temp_dir().join("tamil_tokenizer_repro.txt");
        std::fs::write(&path, REPRO_CORPUS).expect("write corpus");
        let path = path.to_str().expect("utf-8 temp path");

        let (merges_a, vocab_a) = train_once(base_vocab(), path, 40);
        let (merges_b, vocab_b) = train_once(base_vocab(), path, 40);

        assert_eq!(vocab_a, vocab_b, "token ids differ between runs");
        assert_eq!(merges_a, merges_b, "merge order differs between runs");
        let _ = std::fs::remove_file(path);
    }
}
