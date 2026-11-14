// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple model structure for basic inference

use crate::backend::arm::ArmCpuBackend;
use crate::model::GgufFile;
use crate::tokenizer::Tokenizer;
use std::collections::HashMap;

/// Simple model holding weights and config
pub struct SimpleModel {
    /// GGUF file reference
    pub gguf: GgufFile,
    /// Tokenizer
    pub tokenizer: Box<dyn Tokenizer>,
    /// Model config
    pub config: ModelConfig,
    /// CPU backend
    pub backend: ArmCpuBackend,
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layers: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub context_length: usize,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        // Try different architecture prefixes
        let arch = gguf
            .get_meta::<String>("general.architecture")
            .unwrap_or_else(|| "llama".to_string());

        let prefix = format!("{}.", arch);

        Self {
            n_layers: gguf
                .get_meta::<i64>(&format!("{}block_count", prefix))
                .or_else(|| gguf.get_meta::<i64>("llama.block_count"))
                .unwrap_or(32) as usize,
            hidden_size: gguf
                .get_meta::<i64>(&format!("{}embedding_length", prefix))
                .or_else(|| gguf.get_meta::<i64>("llama.embedding_length"))
                .unwrap_or(4096) as usize,
            n_heads: gguf
                .get_meta::<i64>(&format!("{}attention.head_count", prefix))
                .or_else(|| gguf.get_meta::<i64>("llama.attention.head_count"))
                .unwrap_or(32) as usize,
            n_kv_heads: gguf
                .get_meta::<i64>(&format!("{}attention.head_count_kv", prefix))
                .or_else(|| {
                    gguf.get_meta::<i64>("llama.attention.head_count_kv")
                })
                .unwrap_or(32) as usize,
            vocab_size: gguf
                .get_meta::<i64>(&format!("{}vocab_size", prefix))
                .or_else(|| gguf.get_meta::<i64>("llama.vocab_size"))
                .unwrap_or(32000) as usize,
            intermediate_size: gguf
                .get_meta::<i64>(&format!("{}feed_forward_length", prefix))
                .or_else(|| gguf.get_meta::<i64>("llama.feed_forward_length"))
                .unwrap_or(11008) as usize,
            context_length: gguf
                .get_meta::<i64>(&format!("{}context_length", prefix))
                .or_else(|| gguf.get_meta::<i64>("llama.context_length"))
                .unwrap_or(4096) as usize,
            head_dim: 0, // Will be computed
        }
    }

    pub fn head_dim(&self) -> usize {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.hidden_size / self.n_heads
        }
    }
}

impl SimpleModel {
    /// Load model from GGUF file
    pub fn load(
        path: &str,
        tokenizer: Box<dyn Tokenizer>,
    ) -> Result<Self, String> {
        let gguf = GgufFile::load(path)?;
        let config = ModelConfig::from_gguf(&gguf);
        let backend = ArmCpuBackend::new(0);

        println!("Model config: {:?}", config);

        Ok(Self { gguf, tokenizer, config, backend })
    }

    /// Get tokenizer
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Get config
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get tensor data by name
    pub fn get_tensor(&self, name: &str) -> Option<&[u8]> {
        self.gguf.get_tensor_data(name)
    }

    /// Find tensor by prefix search
    pub fn find_tensor(&self, prefix: &str) -> Option<(&String, &[u8])> {
        for tensor in &self.gguf.tensors {
            if tensor.name.starts_with(prefix) || tensor.name.contains(prefix) {
                let data = self.gguf.get_tensor_data(&tensor.name)?;
                return Some((&tensor.name, data));
            }
        }
        None
    }

    /// Get embedding weights
    pub fn get_embeddings(&self) -> Option<&[u8]> {
        // Try different naming conventions
        for name in
            &["token_embd.weight", "embed_tokens.weight", "embed.weight"]
        {
            if let Some(data) = self.gguf.get_tensor_data(name) {
                return Some(data);
            }
        }
        None
    }

    /// Get output layer weights
    pub fn get_output_weight(&self) -> Option<&[u8]> {
        // Try different naming conventions
        for name in &["output.weight", "lm_head.weight", "token_embd.weight"] {
            if let Some(data) = self.gguf.get_tensor_data(name) {
                return Some(data);
            }
        }
        None
    }

    /// Simple token embedding lookup (returns hidden states for a single token)
    pub fn embed_token(&self, _token_id: u32) -> Option<Vec<f32>> {
        let _emb_weight = self.get_embeddings()?;
        let hidden_size = self.config.hidden_size;

        // Assuming F32 or F16 embedding
        // For now, just return zeros as placeholder
        // Real implementation would:
        // 1. Find embedding tensor and its type
        // 2. Extract row for token_id
        // 3. Convert to f32
        Some(vec![0.0f32; hidden_size])
    }

    /// List all tensor names
    pub fn list_tensors(&self) -> Vec<&str> {
        self.gguf.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Print tensor summary
    pub fn print_tensor_summary(&self) {
        // Print first layer tensors
        println!("\nLayer 0 tensors:");
        for tensor in &self.gguf.tensors {
            if tensor.name.starts_with("blk.0.") {
                println!("  {}: {:?}", tensor.name, tensor.quant_type);
            }
        }

        // Print other special tensors
        println!("\nSpecial tensors:");
        for tensor in &self.gguf.tensors {
            if !tensor.name.starts_with("blk.") {
                println!("  {}: {:?}", tensor.name, tensor.quant_type);
            }
        }

        // Group counts
        let mut groups: HashMap<String, usize> = HashMap::new();

        for tensor in &self.gguf.tensors {
            // Extract layer prefix
            let prefix = if tensor.name.starts_with("blk.") {
                let parts: Vec<&str> = tensor.name.split('.').take(2).collect();
                parts.join(".")
            } else if tensor.name.contains('.') {
                tensor.name.split('.').next().unwrap_or("other").to_string()
            } else {
                "other".to_string()
            };

            *groups.entry(prefix).or_insert(0) += 1;
        }

        println!("\nTensor groups:");
        for (prefix, count) in groups.iter() {
            println!("  {}: {} tensors", prefix, count);
        }
    }
}
