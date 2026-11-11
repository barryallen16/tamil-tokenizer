//! Shared vocabulary and merge rules for the Rust encoder and Hugging Face export.
//!
//! Hugging Face's BPE implementation starts from single characters and applies
//! merges in rank order without native seed symbols. This module reconstructs
//! multi-character base clusters from individual characters via promotion merges
//! placed ahead of learned merges, ensuring byte-exact parity across engines.

use crate::vocab::TamilTokenizer;
use rustc_hash::FxHashMap;
use std::collections::HashSet;

/// A vocabulary, its merge list, and the counts the exporter reports.
pub struct CanonicalModel {
    pub vocab: FxHashMap<String, u32>,
    /// Promotion merges first, then the learned merges, in rank order.
    pub merges: Vec<(String, String)>,
    /// How many promotion merges the list starts with.
    pub promotions: usize,
    /// Characters appended for tokens that used a character the vocab lacked.
    pub added_prims: usize,
}

/// Build the model both engines read.
///
/// `vocab` and `merges` are the trained artifact's. The result adds what the
/// trained form cannot express: the characters a token uses but the vocab
/// lacks, the byte tokens, and the promotion merges.
pub fn canonical_model(
    vocab: &FxHashMap<String, u32>,
    merges: &[(String, String)],
) -> CanonicalModel {
    let mut vocab = vocab.clone();
    let (_, byte_decoder) = TamilTokenizer::create_byte_mappings();
    let byte_chars: HashSet<char> = byte_decoder.keys().copied().collect();

    // A character used by a token but absent from the vocab cannot be looked
    // up, so the input cannot be represented. Append those.
    let mut missing: Vec<char> = vocab
        .keys()
        .flat_map(|t| t.chars())
        .filter(|c| !vocab.contains_key(&c.to_string()) && !byte_chars.contains(c))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    let added_prims = missing.len();

    let mut next_id = vocab.values().copied().max().map_or(0, |m| m + 1);
    for c in missing {
        vocab.insert(c.to_string(), next_id);
        next_id += 1;
    }

    // Byte fallback, in Hugging Face's `<0xNN>` spelling. A character the
    // vocab does not hold is emitted as one token per UTF-8 byte, which keeps
    // the input representable instead of dropped.
    for b in 0..=255u8 {
        vocab.insert(format!("<0x{b:02X}>"), next_id);
        next_id += 1;
    }

    let promotions = build_promotions(&vocab, merges, &byte_chars);
    let mut all_merges = promotions.clone();
    all_merges.extend(merges.iter().cloned());

    CanonicalModel {
        vocab,
        merges: all_merges,
        promotions: promotions.len(),
        added_prims,
    }
}

/// Rules that build every multi-character token from single characters.
///
/// Hugging Face's `BPE` model has no seed symbols, so a token like `க்ஷ` is
/// unreachable unless a merge produces it. Each rule here is one step of that
/// reconstruction, and the rules rank ahead of the learned merges.
fn build_promotions(
    vocab: &FxHashMap<String, u32>,
    merges: &[(String, String)],
    byte_chars: &HashSet<char>,
) -> Vec<(String, String)> {
    // A token a merge already produces needs no promotion.
    let produced: HashSet<String> = merges.iter().map(|(a, b)| format!("{a}{b}")).collect();
    let mut reachable: HashSet<String> = byte_chars
        .iter()
        .map(|c| c.to_string())
        .chain(vocab.keys().filter(|t| t.chars().count() == 1).cloned())
        .collect();

    let mut candidates: Vec<&String> = vocab
        .keys()
        .filter(|t| t.chars().count() > 1 && !produced.contains(*t) && !reachable.contains(*t))
        .collect();
    // Longest result first. A promotion's rank decides which merge happens when
    // several are available, and the encoder resolves overlaps by longest
    // match. Emitting the four-character token before the three-character ones
    // is what makes the reconstruction build the longer token instead of
    // stopping at the shorter one.
    candidates.sort_by_key(|t| (std::cmp::Reverse(t.chars().count()), (*t).clone()));

    let mut promotions: Vec<(String, String)> = Vec::new();
    for token in candidates {
        // Split into two vocab tokens, longest prefix first. A flat greedy pass
        // would take the whole token as one part and emit nothing, and a plain
        // prefix pass cannot build ஸ்ரீ, since ஸ்ர is not a token while ஸ் and
        // ரீ both are.
        fn split(
            token: &str,
            vocab: &FxHashMap<String, u32>,
            reachable: &HashSet<String>,
        ) -> Option<Vec<(String, String)>> {
            if reachable.contains(token) {
                return Some(Vec::new());
            }
            let chars: Vec<char> = token.chars().collect();
            for k in (1..chars.len()).rev() {
                let left: String = chars[..k].iter().collect();
                let right: String = chars[k..].iter().collect();
                if !vocab.contains_key(&left) || !vocab.contains_key(&right) {
                    continue;
                }
                let (Some(a), Some(b)) = (
                    split(&left, vocab, reachable),
                    split(&right, vocab, reachable),
                ) else {
                    continue;
                };
                let mut out = a;
                out.extend(b);
                out.push((left, right));
                return Some(out);
            }
            None
        }

        let Some(rules) = split(token, vocab, &reachable) else {
            continue;
        };
        for (a, b) in rules {
            if reachable.insert(format!("{a}{b}")) {
                promotions.push((a, b));
            }
        }
    }

    promotions
}

/// The byte a `<0xNN>` token stands for.
pub fn parse_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}
