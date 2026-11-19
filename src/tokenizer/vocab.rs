// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Vocabulary data structure

use std::collections::HashMap;

use crate::model::gguf::{TokenType, TokenizerMetadata};
use crate::tokenizer::{TokenId, TokenizerError};

// Vocabulary structure with bidirectional mapping
pub struct Vocab {
    // Token ID -> Token string
    id_to_token: Vec<String>,
    // Token string -> Token ID
    token_to_id: HashMap<String, TokenId>,
    // Token types
    token_types: Vec<TokenType>,
    // Token scores (optional, used by SPM)
    scores: Vec<f32>,
    // BPE merge rules: (token1, token2) -> merged_token
    merges: HashMap<(String, String), String>,
    // Merge priority: (token1, token2) -> rank
    merge_ranks: HashMap<(String, String), usize>,
    // Special tokens sorted by length (longest first)
    special_tokens: Vec<TokenId>,
    // BOS token ID
    bos_token_id: Option<TokenId>,
    // EOS token ID
    eos_token_id: Option<TokenId>,
    // UNK token ID
    unk_token_id: Option<TokenId>,
    // PAD token ID
    pad_token_id: Option<TokenId>,
    // Pre-tokenizer type
    pre: Option<String>,
    // Add BOS token during encoding
    pub add_bos: bool,
    // Add EOS token during encoding
    pub add_eos: bool,
    // Byte fallback flag (SPM)
    pub byte_fallback: bool,
}

impl Vocab {
    // Create vocabulary from GGUF metadata
    pub fn new(meta: TokenizerMetadata) -> Result<Self, TokenizerError> {
        if meta.tokens.is_empty() {
            return Err(TokenizerError::InvalidInput(
                "Empty token list".to_string(),
            ));
        }

        // Build id_to_token
        let id_to_token = meta.tokens.clone();

        // Build token_to_id
        let mut token_to_id = HashMap::with_capacity(meta.tokens.len());
        for (id, token) in meta.tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id as TokenId);
        }

        // Build merge ranks (lower rank = higher priority)
        let mut merge_ranks = HashMap::with_capacity(meta.merges.len());
        for (rank, (t1, t2)) in meta.merges.iter().enumerate() {
            merge_ranks.insert((t1.clone(), t2.clone()), rank);
        }

        // Build merges map
        let mut merges = HashMap::with_capacity(meta.merges.len());
        for (t1, t2) in &meta.merges {
            let merged = format!("{}{}", t1, t2);
            merges.insert((t1.clone(), t2.clone()), merged);
        }

        // Collect special tokens (Control, UserDefined, Unknown)
        // Sorted by length (longest first) for greedy matching
        let mut special_tokens: Vec<TokenId> = meta
            .tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| {
                meta.token_types
                    .get(*id)
                    .map(|&t| {
                        t == TokenType::Control
                            || t == TokenType::UserDefined
                            || t == TokenType::Unknown
                    })
                    .unwrap_or(false)
            })
            .map(|(id, _)| id as TokenId)
            .collect();

        // Sort by token length (longest first)
        special_tokens.sort_by(|a, b| {
            let len_a =
                id_to_token.get(*a as usize).map(|s| s.len()).unwrap_or(0);
            let len_b =
                id_to_token.get(*b as usize).map(|s| s.len()).unwrap_or(0);
            len_b.cmp(&len_a)
        });

        Ok(Self {
            id_to_token,
            token_to_id,
            token_types: meta.token_types,
            scores: meta.scores,
            merges,
            merge_ranks,
            special_tokens,
            bos_token_id: meta.bos_token_id,
            eos_token_id: meta.eos_token_id,
            unk_token_id: meta.unk_token_id,
            pad_token_id: meta.pad_token_id,
            pre: meta.pre,
            add_bos: meta.add_bos,
            add_eos: meta.add_eos,
            byte_fallback: meta.byte_fallback,
        })
    }

    // Get token string by ID
    pub fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    // Get token ID by string
    pub fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }

    // Get token score
    pub fn token_score(&self, id: TokenId) -> Option<f32> {
        self.scores.get(id as usize).copied()
    }

    // Get merge rank for a pair of tokens
    pub fn merge_rank(&self, t1: &str, t2: &str) -> Option<usize> {
        self.merge_ranks.get(&(t1.to_string(), t2.to_string())).copied()
    }

    // Check if a merge exists
    pub fn has_merge(&self, t1: &str, t2: &str) -> bool {
        self.merges.contains_key(&(t1.to_string(), t2.to_string()))
    }

    // Get merged token for a pair
    pub fn get_merge(&self, t1: &str, t2: &str) -> Option<&str> {
        self.merges.get(&(t1.to_string(), t2.to_string())).map(|s| s.as_str())
    }

    // Get BOS token ID
    pub fn bos_token(&self) -> Option<TokenId> {
        self.bos_token_id
    }

    // Get EOS token ID
    pub fn eos_token(&self) -> Option<TokenId> {
        self.eos_token_id
    }

    // Get UNK token ID
    pub fn unk_token(&self) -> Option<TokenId> {
        self.unk_token_id
    }

    // Get PAD token ID
    pub fn pad_token(&self) -> Option<TokenId> {
        self.pad_token_id
    }

    // Get vocabulary size
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    // Iterate over all tokens (id, token)
    pub fn tokens_iter(&self) -> impl Iterator<Item = (TokenId, &str)> {
        self.id_to_token
            .iter()
            .enumerate()
            .map(|(id, token)| (id as TokenId, token.as_str()))
    }

    // Get pre-tokenizer type
    pub fn pre(&self) -> Option<&str> {
        self.pre.as_deref()
    }

    // Get all merge pairs sorted by rank
    pub fn merges_by_rank(&self) -> Vec<(&str, &str)> {
        let mut pairs: Vec<_> = self
            .merge_ranks
            .iter()
            .map(|((t1, t2), _)| (t1.as_str(), t2.as_str()))
            .collect();
        pairs.sort_by_key(|(t1, t2)| {
            self.merge_ranks
                .get(&((*t1).to_string(), (*t2).to_string()))
                .copied()
                .unwrap_or(usize::MAX)
        });
        pairs
    }

    // Check if token is a byte token
    pub fn is_byte_token(&self, token: &str) -> bool {
        // Byte tokens are typically <0xNN> format
        if token.starts_with('<') && token.ends_with('>') && token.len() == 6 {
            &token[1..3] == "0x"
                && token[3..5].chars().all(|c| c.is_ascii_hexdigit())
        } else {
            false
        }
    }

    // Get special tokens list (sorted by length, longest first)
    pub fn special_tokens(&self) -> &[TokenId] {
        &self.special_tokens
    }

    // Get token type
    pub fn token_type(&self, id: TokenId) -> TokenType {
        self.token_types.get(id as usize).copied().unwrap_or(TokenType::Normal)
    }

    // Check if token is a special/control token
    pub fn is_special_token(&self, token: &str) -> bool {
        if let Some(&id) = self.token_to_id.get(token) {
            let tt = self.token_type(id);
            tt == TokenType::Control
                || tt == TokenType::UserDefined
                || tt == TokenType::Unknown
        } else {
            false
        }
    }

    // Check if token ID is a special token
    pub fn is_special_token_id(&self, id: TokenId) -> bool {
        let tt = self.token_type(id);
        tt == TokenType::Control
            || tt == TokenType::UserDefined
            || tt == TokenType::Unknown
    }

    // Get byte value from byte token
    pub fn byte_from_token(&self, token: &str) -> Option<u8> {
        if self.is_byte_token(token) {
            u8::from_str_radix(&token[3..5], 16).ok()
        } else {
            None
        }
    }

    // Create byte token for a byte value
    pub fn byte_to_token(&self, byte: u8) -> Option<String> {
        let token = format!("<0x{:02X}>", byte);
        self.token_to_id(&token).map(|_| token)
    }
}
