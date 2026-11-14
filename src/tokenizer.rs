// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Tokenizer module

pub mod bpe;
pub mod vocab;

pub use vocab::Vocab;

use crate::model::gguf::TokenizerMetadata;

// Tokenizer error type
#[derive(Debug, Clone)]
pub enum TokenizerError {
    // Token not found in vocabulary
    UnknownToken(String),
    // Invalid input
    InvalidInput(String),
    // Tokenization failed
    TokenizationFailed(String),
    // Detokenization failed
    DetokenizationFailed(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::UnknownToken(s) => {
                write!(f, "Unknown token: {}", s)
            }
            TokenizerError::InvalidInput(s) => {
                write!(f, "Invalid input: {}", s)
            }
            TokenizerError::TokenizationFailed(s) => {
                write!(f, "Tokenization failed: {}", s)
            }
            TokenizerError::DetokenizationFailed(s) => {
                write!(f, "Detokenization failed: {}", s)
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

// Token ID type
pub type TokenId = u32;

// Tokenizer trait
pub trait Tokenizer: Send + Sync {
    // Encode text to token IDs
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError>;

    // Decode token IDs to text
    fn decode(&self, tokens: &[TokenId]) -> Result<String, TokenizerError>;

    // Get vocabulary size
    fn vocab_size(&self) -> usize;

    // Get token string by ID
    fn id_to_token(&self, id: TokenId) -> Option<&str>;

    // Get token ID by string
    fn token_to_id(&self, token: &str) -> Option<TokenId>;

    // Get BOS token ID
    fn bos_token(&self) -> Option<TokenId>;

    // Get EOS token ID
    fn eos_token(&self) -> Option<TokenId>;

    // Get UNK token ID
    fn unk_token(&self) -> Option<TokenId>;

    // Get PAD token ID
    fn pad_token(&self) -> Option<TokenId>;
}

// Create tokenizer from GGUF metadata
pub fn create_tokenizer(
    meta: TokenizerMetadata,
) -> Result<Box<dyn Tokenizer>, TokenizerError> {
    use crate::model::gguf::TokenizerType;

    match meta.tokenizer_type {
        TokenizerType::BPE => {
            let vocab = Vocab::new(meta)?;
            Ok(Box::new(bpe::BpeTokenizer::new(vocab)))
        }
        TokenizerType::SPM => {
            let vocab = Vocab::new(meta)?;
            Ok(Box::new(bpe::BpeTokenizer::new(vocab)))
        }
        _ => Err(TokenizerError::TokenizationFailed(format!(
            "Unsupported tokenizer type: {:?}",
            meta.tokenizer_type
        ))),
    }
}
