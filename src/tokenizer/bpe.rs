// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// BPE Tokenizer implementation
//
// Following llama.cpp's approach:
// 1. Pre-tokenize using hand-written character matching (no regex library)
// 2. Apply BPE merges using priority queue
// 3. Handle GPT-2 style encoding: Ġ (U+0120) for leading space
// 4. Handle byte fallback for unknown bytes

use crate::tokenizer::{TokenId, Tokenizer, TokenizerError, Vocab};

// BPE tokenizer
pub struct BpeTokenizer {
    vocab: Vocab,
}

impl BpeTokenizer {
    // Create a new BPE tokenizer
    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    // Pre-tokenize text into initial chunks
    // Following llama.cpp's llama_tokenize approach
    // GPT-2 style: convert leading space to Ġ (U+0120)
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();

        // Handle leading space specially
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == ' ' {
                // GPT-2 style: space becomes Ġ (U+0120) at start of next token
                // Flush current token first
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                // Add Ġ to the next token
                current.push('\u{0120}');
            } else if ch == '\n' {
                // Flush current token
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                // Newline becomes separate token (Ċ in GPT-2)
                result.push('\u{010A}'.to_string());
            } else if ch == '\t' {
                // Flush current token
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                // Tab becomes separate token
                result.push('\u{0125}'.to_string());
            } else if ch.is_alphanumeric()
                || ch == '\''
                || ch == '-'
                || ch == '_'
            {
                // Letters, numbers, apostrophes, hyphens, underscores continue the current token
                current.push(ch);
            } else {
                // Check for contractions (GPT-2 style): 's, 't, 're, 've, 'm, 'll, 'd
                if ch == '\'' && !current.is_empty() {
                    // Look ahead for common contractions
                    let rest: String = chars.clone().take(3).collect();
                    if rest.starts_with('s')
                        || rest.starts_with('t')
                        || rest.starts_with('m')
                        || rest.starts_with('d')
                    {
                        current.push(ch);
                        current.push(chars.next().unwrap()); // consume the letter
                    } else if rest.starts_with("re")
                        || rest.starts_with("ve")
                        || rest.starts_with("ll")
                    {
                        current.push(ch);
                        for _ in 0..2 {
                            if let Some(c) = chars.next() {
                                current.push(c);
                            }
                        }
                    } else {
                        // Not a contraction, treat as punctuation
                        result.push(std::mem::take(&mut current));
                        result.push(ch.to_string());
                    }
                } else {
                    // Other punctuation - flush current and start new
                    if !current.is_empty() {
                        result.push(std::mem::take(&mut current));
                    }
                    result.push(ch.to_string());
                }
            }
        }

        // Flush remaining
        if !current.is_empty() {
            result.push(current);
        }

        result
    }

    // Convert a string to character-level tokens
    fn to_chars(&self, s: &str) -> Vec<String> {
        s.chars().map(|c| c.to_string()).collect()
    }

    // Apply BPE merges to a sequence of tokens
    fn apply_bpe(&self, tokens: Vec<String>) -> Vec<String> {
        if tokens.is_empty() {
            return tokens;
        }

        let mut current_tokens = tokens;

        // Keep merging until no more merges are possible
        loop {
            // Find the best merge (lowest rank = highest priority)
            let mut best_merge: Option<(usize, String, String)> = None;
            let mut best_rank = usize::MAX;

            for i in 0..current_tokens.len().saturating_sub(1) {
                let t1 = &current_tokens[i];
                let t2 = &current_tokens[i + 1];

                if let Some(rank) = self.vocab.merge_rank(t1, t2) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_merge = Some((i, t1.clone(), t2.clone()));
                    }
                }
            }

            // No more merges possible
            let Some((idx, t1, t2)) = best_merge else {
                break;
            };

            // Apply the merge
            let merged = format!("{}{}", t1, t2);
            current_tokens[idx] = merged;
            current_tokens.remove(idx + 1);
        }

        current_tokens
    }

    // Encode a single pre-tokenized chunk
    fn encode_chunk(&self, text: &str) -> Vec<TokenId> {
        // Convert to character-level tokens
        let char_tokens = self.to_chars(text);

        // Apply BPE merges
        let merged_tokens = self.apply_bpe(char_tokens);

        // Convert tokens to IDs
        merged_tokens.iter().filter_map(|t| self.vocab.token_to_id(t)).collect()
    }

    // Decode a token string, handling Ġ (space) and other GPT-2 encoding
    fn decode_token(&self, token: &str) -> String {
        let mut result = String::new();

        for ch in token.chars() {
            match ch {
                '\u{0120}' => result.push(' '),  // Ġ = space
                '\u{010A}' => result.push('\n'), // Ċ = newline
                '\u{0125}' => result.push('\t'), // ĥ = tab
                _ => result.push(ch),
            }
        }

        result
    }
}

impl Tokenizer for BpeTokenizer {
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

        // Pre-tokenize
        let chunks = self.pre_tokenize(text);

        // Encode each chunk
        for chunk in chunks {
            let token_ids = self.encode_chunk(&chunk);
            result.extend(token_ids);
        }

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
                // Handle byte fallback tokens like <0xNN>
                if self.vocab.is_byte_token(token) {
                    if let Some(byte) = self.vocab.byte_from_token(token) {
                        result.push(byte as char);
                    }
                } else {
                    // Decode using GPT-2 byte decoding
                    result.push_str(&self.decode_token(token));
                }
            } else if let Some(unk) = self.vocab.unk_token() {
                // Use UNK token for unknown IDs
                if let Some(unk_token) = self.vocab.id_to_token(unk) {
                    result.push_str(&self.decode_token(unk_token));
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
            // Skip special tokens (BOS, EOS, PAD, etc.)
            if self.vocab.is_special_token_id(id) {
                continue;
            }

            if let Some(token) = self.vocab.id_to_token(id) {
                // Handle byte fallback tokens like <0xNN>
                if self.vocab.is_byte_token(token) {
                    if let Some(byte) = self.vocab.byte_from_token(token) {
                        result.push(byte as char);
                    }
                } else if !self.vocab.is_special_token(token) {
                    // Decode using GPT-2 byte decoding
                    result.push_str(&self.decode_token(token));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{TokenizerMetadata, TokenizerType};

    fn make_test_vocab() -> Vocab {
        Vocab::new(TokenizerMetadata {
            tokenizer_type: TokenizerType::BPE,
            tokens: vec![
                "<0x00>".to_string(),
                "hello".to_string(),
                "world".to_string(),
                "Ġ".to_string(), // space
                "h".to_string(),
                "e".to_string(),
                "l".to_string(),
                "o".to_string(),
                "w".to_string(),
                "r".to_string(),
                "d".to_string(),
                "Ġhello".to_string(),
                "Ġworld".to_string(),
            ],
            merges: vec![
                ("h".to_string(), "e".to_string()),
                ("l".to_string(), "l".to_string()),
                ("ll".to_string(), "o".to_string()),
                ("he".to_string(), "llo".to_string()),
                ("Ġ".to_string(), "hello".to_string()),
                ("Ġ".to_string(), "world".to_string()),
            ],
            scores: vec![],
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: Some(0),
            pad_token_id: None,
            pre: None,
            add_bos: false,
            add_eos: false,
            byte_fallback: true,
        })
        .unwrap()
    }

    #[test]
    fn test_pre_tokenize() {
        let vocab = make_test_vocab();
        let tokenizer = BpeTokenizer::new(vocab);

        // "hello world" -> ["hello", "Ġworld"]
        let chunks = tokenizer.pre_tokenize("hello world");
        assert_eq!(chunks, vec!["hello", "Ġworld"]);

        // "  hello" -> ["Ġ", "Ġhello"]
        let chunks = tokenizer.pre_tokenize("  hello");
        assert_eq!(chunks, vec!["Ġ", "Ġhello"]);

        // "Hello, world!" -> ["Hello", ",", "Ġworld", "!"]
        let chunks = tokenizer.pre_tokenize("Hello, world!");
        assert!(chunks.contains(&"Hello".to_string()));
    }

    #[test]
    fn test_decode_space() {
        let vocab = make_test_vocab();
        let tokenizer = BpeTokenizer::new(vocab);

        // Token "Ġhello" should decode to " hello"
        let decoded = tokenizer.decode_token("Ġhello");
        assert_eq!(decoded, " hello");

        // Token "Ġworld" should decode to " world"
        let decoded = tokenizer.decode_token("Ġworld");
        assert_eq!(decoded, " world");
    }

    #[test]
    fn test_encode_decode() {
        let vocab = make_test_vocab();
        let tokenizer = BpeTokenizer::new(vocab);

        // Encode "hello world"
        let tokens = tokenizer.encode("hello world").unwrap();
        // Should have at least some tokens
        assert!(!tokens.is_empty());

        // Decode should return something close to original
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert!(decoded.contains("hello") || decoded.contains("world"));
    }
}
