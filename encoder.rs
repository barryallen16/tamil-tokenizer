use std::sync::Arc;
use rustc_hash::FxHashMap;
use crate::pretokenizer::TamilPretokenizer;
use crate::vocab::Result;

#[allow(dead_code)]
pub struct FastBPEEncoder {
    token_to_id: Arc<FxHashMap<String, u32>>,
    token_to_id_bytes: Arc<FxHashMap<Vec<u8>, u32>>,
    id_to_token: Arc<Vec<String>>,
    pretokenizer: TamilPretokenizer,
    byte_to_id: [Option<u32>; 256],
}

impl FastBPEEncoder {
    pub fn new(vocab: FxHashMap<String, u32>) -> Result<Self> {
        let mut id_to_token = vec![String::new(); vocab.len()];
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
        }

        let mut byte_to_id = [None; 256];
        for (token, &id) in &vocab {
            if token.len() == 1 {
                byte_to_id[token.as_bytes()[0] as usize] = Some(id);
            }
        }

        let token_to_id_bytes: FxHashMap<Vec<u8>, u32> = vocab
            .iter()
            .map(|(token, &id)| (token.as_bytes().to_vec(), id))
            .collect();

        Ok(Self {
            token_to_id: Arc::new(vocab),
            token_to_id_bytes: Arc::new(token_to_id_bytes),
            id_to_token: Arc::new(id_to_token),
            pretokenizer: TamilPretokenizer::new(),
            byte_to_id,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut token_ids = Vec::with_capacity(text.len() / 3);
        let chunks = self.pretokenizer.pretokenize(text);

        for chunk in chunks {
            let bytes = chunk.as_bytes();
            let mut pos = 0;

            while pos < bytes.len() {
                // longest match against the vocabulary
                let mut matched = false;
                for len in (1..=bytes.len() - pos).rev() {
                    if let Some(&id) = self.token_to_id_bytes.get(&bytes[pos..pos + len]) {
                        token_ids.push(id);
                        pos += len;
                        matched = true;
                        break;
                    }
                }

                if !matched {
                    if let Some(id) = self.byte_to_id[bytes[pos] as usize] {
                        token_ids.push(id);
                    }
                    pos += 1;
                }
            }
        }

        token_ids
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .map(|s| s.as_str())
            .collect::<String>()
    }
}