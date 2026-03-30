//! Thin wrapper around HuggingFace's `tokenizers` crate.
//! Loads `tokenizer.json` from the model directory.

use std::path::Path;
use tokenizers::Tokenizer;

pub struct TokenizerWrapper {
    inner: Tokenizer,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl TokenizerWrapper {
    /// Load a `tokenizer.json` file from the given path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // Common defaults for Llama-family models
        let bos_token_id = tokenizer
            .token_to_id("<|begin_of_text|>")
            .or_else(|| tokenizer.token_to_id("<s>"))
            .unwrap_or(1);

        let eos_token_id = tokenizer
            .token_to_id("<|end_of_text|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        Ok(Self { inner: tokenizer, eos_token_id, bos_token_id })
    }

    /// Encode text into token IDs. Prepends BOS if `add_bos` is true.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, String> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| format!("Encode error: {}", e))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if add_bos {
            ids.insert(0, self.bos_token_id);
        }
        Ok(ids)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner.decode(ids, true)
            .map_err(|e| format!("Decode error: {}", e))
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> String {
        self.inner.decode(&[id], true).unwrap_or_default()
    }
}
