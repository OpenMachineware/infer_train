// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// BPE Tokenizer for GGUF models

use crate::model::gguf::{GgufFile, MetadataValue};
use std::collections::HashMap;

/// Pre-tokenization type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    Default,
    Llama3,
    DeepSeekLlm,
    DeepSeekCoder,
    Gpt2,
    Qwen2,
    Qwen35,
    Tekken,
    ChatGlm3,
    ChatGlm4,
    Starcoder,
    CommandR,
    StableLm2,
    Unknown,
}

impl PreType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "default" | "llama-bpe" => PreType::Default,
            "llama3" => PreType::Llama3,
            "deepseek-llm" => PreType::DeepSeekLlm,
            "deepseek-coder" => PreType::DeepSeekCoder,
            "gpt2" => PreType::Gpt2,
            "qwen2" => PreType::Qwen2,
            "qwen35" => PreType::Qwen35,
            "tekken" => PreType::Tekken,
            "chatglm3" => PreType::ChatGlm3,
            "chatglm4" => PreType::ChatGlm4,
            "starcoder" => PreType::Starcoder,
            "command-r" => PreType::CommandR,
            "stablelm2" => PreType::StableLm2,
            _ => PreType::Unknown,
        }
    }
}

/// BPE Tokenizer loaded from GGUF metadata
pub struct BpeTokenizer {
    /// Token string -> ID
    vocab: HashMap<String, u32>,
    /// ID -> Token string
    id_to_token: Vec<String>,
    /// Merge rules: (token_a, token_b) -> priority rank
    merges: HashMap<(String, String), usize>,
    /// Special tokens
    pub bos_token: u32,
    pub eos_token: u32,
    pub unk_token: u32,
    pub pad_token: u32,
    /// Pre-tokenization type
    pub pre_type: PreType,
    /// Config
    pub add_bos: bool,
    pub add_eos: bool,
}

impl BpeTokenizer {
    /// Create tokenizer from GGUF metadata arrays
    pub fn new(
        tokens: &[String],
        merges: &[String],
        bos_token_id: u32,
        eos_token_id: u32,
        unk_token_id: u32,
        pad_token_id: u32,
        add_bos: bool,
        add_eos: bool,
        pre_type: PreType,
    ) -> Self {
        // Build vocab mapping
        let mut vocab = HashMap::new();
        let mut id_to_token = Vec::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
            id_to_token.push(token.clone());
        }

        // Build merge rules (lower index = higher priority)
        let mut merge_map = HashMap::new();
        for (rank, merge) in merges.iter().enumerate() {
            // Merge format: "token_a token_b"
            if let Some(space_pos) = merge.find(' ') {
                let a = merge[..space_pos].to_string();
                let b = merge[space_pos + 1..].to_string();
                merge_map.insert((a, b), rank);
            }
        }

        BpeTokenizer {
            vocab,
            id_to_token,
            merges: merge_map,
            bos_token: bos_token_id,
            eos_token: eos_token_id,
            unk_token: unk_token_id,
            pad_token: pad_token_id,
            pre_type,
            add_bos,
            add_eos,
        }
    }

    /// Load tokenizer from GGUF file
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        let meta = &gguf.metadata;

        // Extract tokens array
        let tokens: Vec<String> = match meta.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| {
                    if let MetadataValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => return None,
        };

        // Extract merges array
        let merges: Vec<String> = match meta.get("tokenizer.ggml.merges") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| {
                    if let MetadataValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => return None,
        };

        // Extract special token IDs
        // BOS token may not be present for all models (e.g., Qwen)
        let bos_token_id =
            get_u32(meta, "tokenizer.ggml.bos_token_id").unwrap_or(0);
        let eos_token_id = get_u32(meta, "tokenizer.ggml.eos_token_id")?;
        let unk_token_id =
            get_u32(meta, "tokenizer.ggml.unknown_token_id").unwrap_or(0);
        let pad_token_id =
            get_u32(meta, "tokenizer.ggml.padding_token_id").unwrap_or(0);

        // Extract config
        let add_bos =
            get_bool(meta, "tokenizer.ggml.add_bos_token").unwrap_or(true);
        let add_eos =
            get_bool(meta, "tokenizer.ggml.add_eos_token").unwrap_or(false);

        // Extract pre-tokenization type
        let pre_type = match meta.get("tokenizer.ggml.pre") {
            Some(MetadataValue::String(s)) => PreType::from_str(s),
            _ => PreType::Default,
        };

        Some(BpeTokenizer::new(
            &tokens,
            &merges,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            pad_token_id,
            add_bos,
            add_eos,
            pre_type,
        ))
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Apply pre-tokenization to split text into chunks
        let chunks = self.pre_tokenize(text);

        let mut all_ids = Vec::new();

        for chunk in chunks {
            // Convert to GPT-2 byte encoding
            let bytes = self.bytes_to_gpt2(&chunk);

            // Split into initial tokens (characters in GPT-2 encoding)
            let tokens = self.bpe_encode_optimized(&bytes);

            // Convert to IDs
            for t in tokens {
                let id = self.vocab.get(&t).copied().unwrap_or(self.unk_token);
                all_ids.push(id);
            }
        }

        // Add special tokens
        if self.add_bos {
            all_ids.insert(0, self.bos_token);
        }
        if self.add_eos {
            all_ids.push(self.eos_token);
        }

        all_ids
    }

    /// Optimized BPE encoding using priority queue - O(n log n)
    fn bpe_encode_optimized(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 1 {
            return vec![text.to_string()];
        }

        // Use linked list + priority queue for O(n log n) complexity
        // Each node: (token_string, prev_index, next_index)
        // prev_index and next_index are -1 if none
        let mut tokens: Vec<(String, i32, i32)> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let prev = if i > 0 { (i - 1) as i32 } else { -1 };
                let next = if i < n - 1 { (i + 1) as i32 } else { -1 };
                (c.to_string(), prev, next)
            })
            .collect();

        // Track which positions are still alive
        let mut alive: Vec<bool> = vec![true; n];

        // Min-heap: (rank, left_position)
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();

        // Initialize heap with all mergeable pairs
        for i in 0..n - 1 {
            let key = (tokens[i].0.clone(), tokens[i + 1].0.clone());
            if let Some(&rank) = self.merges.get(&key) {
                heap.push(Reverse((rank, i)));
            }
        }

        // Process merges
        while let Some(Reverse((rank, pos))) = heap.pop() {
            // Check if left position is still alive
            if !alive[pos] {
                continue;
            }

            // Get next position from linked list
            let next_pos = tokens[pos].2;
            if next_pos == -1 || !alive[next_pos as usize] {
                continue;
            }
            let next_idx = next_pos as usize;

            // Verify the pair still matches (tokens haven't changed)
            let key = (tokens[pos].0.clone(), tokens[next_idx].0.clone());
            match self.merges.get(&key) {
                Some(&r) if r == rank => {} // Valid merge
                Some(&r) => {
                    // Rank changed, re-add with correct rank
                    heap.push(Reverse((r, pos)));
                    continue;
                }
                None => continue, // Pair no longer mergeable
            }

            // Apply merge at position pos
            let merged = tokens[pos].0.clone() + &tokens[next_idx].0;
            let new_next = tokens[next_idx].2;

            tokens[pos].0 = merged;
            tokens[pos].2 = new_next;

            // Mark merged node as dead
            alive[next_idx] = false;

            // Update prev pointer of new_next if exists
            if new_next != -1 {
                tokens[new_next as usize].1 = pos as i32;
            }

            // Add new pairs formed with neighbors
            // Left neighbor
            let left_pos = tokens[pos].1;
            if left_pos != -1 && alive[left_pos as usize] {
                let left_idx = left_pos as usize;
                let key = (tokens[left_idx].0.clone(), tokens[pos].0.clone());
                if let Some(&r) = self.merges.get(&key) {
                    heap.push(Reverse((r, left_idx)));
                }
            }

            // Right neighbor
            let right_pos = tokens[pos].2;
            if right_pos != -1 && alive[right_pos as usize] {
                let right_idx = right_pos as usize;
                let key = (tokens[pos].0.clone(), tokens[right_idx].0.clone());
                if let Some(&r) = self.merges.get(&key) {
                    heap.push(Reverse((r, pos)));
                }
            }
        }

        // Collect surviving tokens in order
        // Find the first alive position
        let mut result = Vec::new();
        let mut current = 0i32;
        while current != -1 && (current as usize) < n {
            if alive[current as usize] {
                result.push(tokens[current as usize].0.clone());
                // Move to next in linked list
                current = tokens[current as usize].2;
            } else {
                // Skip dead nodes by finding next alive
                // This shouldn't happen if we track correctly, but handle it
                current = tokens[current as usize].2;
            }
        }

        result
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                bytes.push_str(token);
            }
        }
        // Convert from GPT-2 byte encoding back to UTF-8
        self.gpt2_to_bytes(&bytes)
    }

    /// Pre-tokenize text into chunks based on pre_type
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match self.pre_type {
            PreType::Qwen35 | PreType::Qwen2 => {
                // Qwen uses regex-based pre-tokenization
                // Simplified pattern without lookahead
                use once_cell::sync::Lazy;
                use regex::Regex;

                static QWEN_REGEX: Lazy<Regex> = Lazy::new(|| {
                    Regex::new(
                        r"(?:'[sdmt]|'ll|'ve|'re)|(?:[^\r\n\p{L}\p{N}]?\p{L}+)|(?:\p{N}{1,3})|(?:[^\s\p{L}\p{N}]+)|(?:\s+)",
                    )
                    .unwrap()
                });

                QWEN_REGEX
                    .find_iter(text)
                    .map(|m| m.as_str().to_string())
                    .collect()
            }
            PreType::Tekken => {
                // Tekken pre-tokenization
                use once_cell::sync::Lazy;
                use regex::Regex;

                static TEKKEN_REGEX: Lazy<Regex> = Lazy::new(|| {
                    Regex::new(r"(?:'[sdmt]|'ll|'ve|'re)|(?: ?\p{L}+)|(?: ?\p{N}+)|(?: ?[^\s\p{L}\p{N}]+)|(?:\s+)").unwrap()
                });

                TEKKEN_REGEX
                    .find_iter(text)
                    .map(|m| m.as_str().to_string())
                    .collect()
            }
            PreType::Llama3 => {
                // Llama3 pre-tokenization
                use once_cell::sync::Lazy;
                use regex::Regex;

                static LLAMA3_REGEX: Lazy<Regex> = Lazy::new(|| {
                    Regex::new(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\p{L}\p{N}]?\p{L}+)|(?:\p{N}{1,3})|(?: ?[^\s\p{L}\p{N}]+)|(?:\s+)").unwrap()
                });

                LLAMA3_REGEX
                    .find_iter(text)
                    .map(|m| m.as_str().to_string())
                    .collect()
            }
            _ => {
                // Default: treat entire text as single chunk
                vec![text.to_string()]
            }
        }
    }

    /// Convert text to GPT-2 byte encoding
    /// GPT-2 maps bytes to unicode characters to avoid control characters
    pub fn bytes_to_gpt2(&self, text: &str) -> String {
        // GPT-2 byte-to-unicode mapping
        // Bytes 33-126 (printable ASCII except space) map to themselves
        // Bytes 161-172 and 174-255 map to themselves
        // Other bytes (0-32, 127-160, 173) map to unicode chars starting at 256

        static BYTE_ENCODER: once_cell::sync::Lazy<Vec<char>> =
            once_cell::sync::Lazy::new(|| {
                let mut map = vec!['\0'; 256];
                let mut unicode = 256u32;

                for byte in 0u8..=255u8 {
                    let ch = if (byte >= 33 && byte <= 126)
                        || (byte >= 161 && byte <= 172)
                        || byte >= 174
                    {
                        byte as char
                    } else {
                        let c = char::from_u32(unicode).unwrap();
                        unicode += 1;
                        c
                    };
                    map[byte as usize] = ch;
                }
                map
            });

        text.bytes().map(|b| BYTE_ENCODER[b as usize]).collect()
    }

    /// Convert GPT-2 byte encoding back to bytes
    fn gpt2_to_bytes(&self, text: &str) -> String {
        // Reverse mapping
        static BYTE_DECODER: once_cell::sync::Lazy<HashMap<char, u8>> =
            once_cell::sync::Lazy::new(|| {
                let mut map = HashMap::new();
                let mut unicode = 256u32;

                for byte in 0u8..=255u8 {
                    let ch = if (byte >= 33 && byte <= 126)
                        || (byte >= 161 && byte <= 172)
                        || byte >= 174
                    {
                        byte as char
                    } else {
                        let c = char::from_u32(unicode).unwrap();
                        unicode += 1;
                        c
                    };
                    map.insert(ch, byte);
                }
                map
            });

        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| BYTE_DECODER.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Get token string by ID
    pub fn token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if token exists in vocab
    pub fn vocab_contains(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Get merge rank for a pair (for debugging)
    pub fn get_merge_rank(&self, a: &str, b: &str) -> Option<usize> {
        self.merges.get(&(a.to_string(), b.to_string())).copied()
    }
}

// Helper functions for extracting metadata
fn get_u32(meta: &HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    match meta.get(key) {
        Some(MetadataValue::Uint32(v)) => Some(*v),
        Some(MetadataValue::Int32(v)) => Some(*v as u32),
        Some(MetadataValue::Uint64(v)) => Some(*v as u32),
        Some(MetadataValue::Int64(v)) => Some(*v as u32),
        _ => None,
    }
}

fn get_bool(meta: &HashMap<String, MetadataValue>, key: &str) -> Option<bool> {
    match meta.get(key) {
        Some(MetadataValue::Bool(v)) => Some(*v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_gpt2() {
        let tokenizer = BpeTokenizer::new(
            &["test".to_string()],
            &[],
            1,
            2,
            0,
            0,
            false,
            false,
            PreType::Default,
        );

        // Space should become 'Ġ' (unicode 288 in GPT-2 encoding)
        let encoded = tokenizer.bytes_to_gpt2(" hello");
        assert!(encoded.starts_with("Ġ"));
    }
}
