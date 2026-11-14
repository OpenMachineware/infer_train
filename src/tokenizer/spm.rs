// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// SPM (SentencePiece) Tokenizer implementation
//
// Following llama.cpp's approach:
// 1. Use trie structure for fast token lookup
// 2. Use scores for priority (lower = higher priority)
// 3. Support byte fallback for unknown characters
// 4. Different merging algorithm than BPE

use crate::tokenizer::{TokenId, Tokenizer, TokenizerError, Vocab};

// Trie node for fast prefix matching
struct TrieNode {
    children: std::collections::HashMap<char, TrieNode>,
    token_id: Option<TokenId>,
    score: f32,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: std::collections::HashMap::new(),
            token_id: None,
            score: f32::MAX,
        }
    }

    fn insert(&mut self, token: &str, id: TokenId, score: f32) {
        let mut node = self;
        for ch in token.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        node.token_id = Some(id);
        node.score = score;
    }

    fn find(&self, text: &str) -> Option<(TokenId, f32, usize)> {
        let mut node = self;
        let mut result: Option<(TokenId, f32, usize)> = None;
        let mut len = 0;

        for ch in text.chars() {
            match node.children.get(&ch) {
                Some(next) => {
                    node = next;
                    len += ch.len_utf8();
                    if let Some(id) = node.token_id {
                        // Prefer lower score (higher priority)
                        if result.is_none() || node.score < result.unwrap().1 {
                            result = Some((id, node.score, len));
                        }
                    }
                }
                None => break,
            }
        }

        result
    }
}

// SPM tokenizer
pub struct SpmTokenizer {
    vocab: Vocab,
    trie: TrieNode,
    #[allow(dead_code)]
    byte_fallback_trie: TrieNode,
}

impl SpmTokenizer {
    // Create a new SPM tokenizer
    pub fn new(vocab: Vocab) -> Self {
        let mut trie = TrieNode::new();
        let mut byte_fallback_trie = TrieNode::new();

        // Build trie from vocabulary
        for (id, token) in vocab.tokens_iter() {
            let score = vocab.token_score(id).unwrap_or(f32::MAX);

            // Check if it's a byte fallback token
            if vocab.is_byte_token(token) {
                if let Some(byte) = vocab.byte_from_token(token) {
                    // Insert into byte fallback trie
                    byte_fallback_trie.insert(token, id, score);
                }
            } else {
                trie.insert(token, id, score);
            }
        }

        Self { vocab, trie, byte_fallback_trie }
    }

    // Encode text using greedy longest-match-first with scores
    fn encode_greedy(&self, text: &str) -> Vec<TokenId> {
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Try to find the longest matching token
            if let Some((id, _, len)) = self.trie.find(remaining) {
                result.push(id);
                remaining = &remaining[len..];
            } else {
                // No match found, try byte fallback
                let ch = remaining.chars().next().unwrap();
                let ch_len = ch.len_utf8();

                // Try byte fallback for the first byte
                let byte_token = format!("<0x{:02X}>", ch as u8);
                if let Some(id) = self.vocab.token_to_id(&byte_token) {
                    result.push(id);
                } else if let Some(unk) = self.vocab.unk_token() {
                    // Use UNK token
                    result.push(unk);
                }

                remaining = &remaining[ch_len..];
            }
        }

        result
    }
}

impl Tokenizer for SpmTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();

        // Add BOS token if configured
        if self.vocab.add_bos {
            if let Some(bos) = self.vocab.bos_token() {
                result.push(bos);
            }
        }

        // Encode using greedy algorithm
        let tokens = self.encode_greedy(text);
        result.extend(tokens);

        // Add EOS token if configured
        if self.vocab.add_eos {
            if let Some(eos) = self.vocab.eos_token() {
                result.push(eos);
            }
        }

        Ok(result)
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String, TokenizerError> {
        let mut result = String::new();

        for &id in tokens {
            if let Some(token) = self.vocab.id_to_token(id) {
                // Handle byte fallback tokens
                if self.vocab.is_byte_token(token) {
                    if let Some(byte) = self.vocab.byte_from_token(token) {
                        result.push(byte as char);
                    }
                } else {
                    result.push_str(token);
                }
            } else if let Some(unk) = self.vocab.unk_token() {
                if let Some(unk_token) = self.vocab.id_to_token(unk) {
                    result.push_str(unk_token);
                }
            }
        }

        Ok(result)
    }

    fn decode_skip_special(
        &self,
        tokens: &[TokenId],
    ) -> Result<String, TokenizerError> {
        let mut result = String::new();

        for &id in tokens {
            // Skip special tokens
            if self.vocab.is_special_token_id(id) {
                continue;
            }

            if let Some(token) = self.vocab.id_to_token(id) {
                if self.vocab.is_byte_token(token) {
                    if let Some(byte) = self.vocab.byte_from_token(token) {
                        result.push(byte as char);
                    }
                } else if !self.vocab.is_special_token(token) {
                    result.push_str(token);
                }
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.vocab.id_to_token(id)
    }

    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.token_to_id(token)
    }

    fn bos_token(&self) -> Option<TokenId> {
        self.vocab.bos_token()
    }

    fn eos_token(&self) -> Option<TokenId> {
        self.vocab.eos_token()
    }

    fn unk_token(&self) -> Option<TokenId> {
        self.vocab.unk_token()
    }

    fn pad_token(&self) -> Option<TokenId> {
        self.vocab.pad_token()
    }
}
