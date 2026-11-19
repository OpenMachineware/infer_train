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
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        // Try different architecture prefixes
        let arch = gguf
            .get_meta::<String>("general.architecture")
            .unwrap_or_else(|| "llama".to_string());

        let prefix = format!("{}.", arch);

        // Try to get vocab_size from metadata, fallback to token_embd dimensions
        let vocab_size = gguf
            .get_meta::<u64>(&format!("{}vocab_size", prefix))
            .or_else(|| gguf.get_meta::<u64>("llama.vocab_size"))
            .or_else(|| {
                // Fallback: infer from token_embd.weight dimensions
                // token_embd.weight shape is [hidden_size, vocab_size]
                gguf.tensors
                    .iter()
                    .find(|t| t.name == "token_embd.weight")
                    .and_then(|t| t.dims.get(1).copied())
            })
            .unwrap_or(32000) as usize;

        Self {
            n_layers: gguf
                .get_meta::<u64>(&format!("{}block_count", prefix))
                .or_else(|| gguf.get_meta::<u64>("llama.block_count"))
                .unwrap_or(32) as usize,
            hidden_size: gguf
                .get_meta::<u64>(&format!("{}embedding_length", prefix))
                .or_else(|| gguf.get_meta::<u64>("llama.embedding_length"))
                .unwrap_or(4096) as usize,
            n_heads: gguf
                .get_meta::<u64>(&format!("{}attention.head_count", prefix))
                .or_else(|| gguf.get_meta::<u64>("llama.attention.head_count"))
                .unwrap_or(32) as usize,
            n_kv_heads: gguf
                .get_meta::<u64>(&format!("{}attention.head_count_kv", prefix))
                .or_else(|| {
                    gguf.get_meta::<u64>("llama.attention.head_count_kv")
                })
                .unwrap_or(32) as usize,
            head_dim: gguf
                .get_meta::<u64>(&format!("{}attention.key_length", prefix))
                .or_else(|| gguf.get_meta::<u64>("llama.attention.key_length"))
                .or_else(|| {
                    gguf.get_meta::<u64>(&format!(
                        "{}attention.head_dim",
                        prefix
                    ))
                })
                .or_else(|| gguf.get_meta::<u64>("llama.attention.head_dim"))
                .unwrap_or(0) as usize,
            vocab_size,
            intermediate_size: gguf
                .get_meta::<u64>(&format!("{}feed_forward_length", prefix))
                .or_else(|| gguf.get_meta::<u64>("llama.feed_forward_length"))
                .unwrap_or(11008) as usize,
            context_length: gguf
                .get_meta::<u64>(&format!("{}context_length", prefix))
                .or_else(|| gguf.get_meta::<u64>("llama.context_length"))
                .unwrap_or(4096) as usize,
            rope_freq_base: gguf
                .get_meta::<f32>(&format!("{}rope.freq_base", prefix))
                .or_else(|| gguf.get_meta::<f32>("llama.rope.freq_base"))
                .or_else(|| {
                    gguf.get_meta::<f32>(&format!("{}rope_base", prefix))
                })
                .or_else(|| gguf.get_meta::<f32>("llama.rope_base"))
                .unwrap_or(10000.0),
            rms_norm_eps: gguf
                .get_meta::<f32>(&format!(
                    "{}attention.layer_norm_rms_epsilon",
                    prefix
                ))
                .or_else(|| {
                    gguf.get_meta::<f32>(
                        "llama.attention.layer_norm_rms_epsilon",
                    )
                })
                .or_else(|| {
                    gguf.get_meta::<f32>(&format!(
                        "{}attention.layer_norm_epsilon",
                        prefix
                    ))
                })
                .or_else(|| {
                    gguf.get_meta::<f32>("llama.attention.layer_norm_epsilon")
                })
                .unwrap_or(1e-6),
        }
    }

    pub fn head_dim(&self) -> usize {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            // Fallback: infer from Q weight shape
            // This should not happen if GGUF has key_length metadata
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

    /// Token embedding lookup (returns hidden states for a single token)
    pub fn embed_token(&self, token_id: u32) -> Option<Vec<f32>> {
        let emb_data = self.get_embeddings()?;
        let hidden_size = self.config.hidden_size;

        // Find embedding tensor info to get quantization type
        let emb_tensor =
            self.gguf.tensors.iter().find(|t| t.name == "token_embd.weight")?;

        match emb_tensor.quant_type {
            crate::types::QuantType::F32 => {
                let offset = (token_id as usize) * hidden_size * 4;
                if offset + hidden_size * 4 > emb_data.len() {
                    return None;
                }
                let bytes = &emb_data[offset..offset + hidden_size * 4];
                Some(
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| {
                            f32::from_le_bytes(chunk.try_into().unwrap())
                        })
                        .collect(),
                )
            }
            crate::types::QuantType::Q6_K => {
                use crate::types::{BlockQ6_K, QK_K};
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let block_size = std::mem::size_of::<BlockQ6_K>();
                let row_offset = (token_id as usize) * nb * block_size;

                if row_offset + nb * block_size > emb_data.len() {
                    return None;
                }

                let mut embedding = vec![0.0f32; hidden_size];
                let block_data =
                    &emb_data[row_offset..row_offset + nb * block_size];

                for i in 0..nb {
                    let block_offset = i * block_size;
                    let block_bytes =
                        &block_data[block_offset..block_offset + block_size];
                    crate::backend::arm::dequant::dequant_q6_k(
                        block_bytes,
                        &mut embedding[i * QK_K..],
                        QK_K.min(hidden_size - i * QK_K),
                    );
                }

                Some(embedding)
            }
            _ => None, // Unsupported quantization type
        }
    }

    /// List all tensor names
    pub fn list_tensors(&self) -> Vec<&str> {
        self.gguf.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Print tensor summary
    pub fn print_tensor_summary(&self) {
        // Print first layer tensors with dimensions
        println!("\nLayer 0 tensors:");
        for tensor in &self.gguf.tensors {
            if tensor.name.starts_with("blk.0.") {
                println!(
                    "  {}: {:?}, dims={:?}",
                    tensor.name, tensor.quant_type, tensor.dims
                );
            }
        }

        // Print other special tensors with dimensions
        println!("\nSpecial tensors:");
        for tensor in &self.gguf.tensors {
            if !tensor.name.starts_with("blk.") {
                println!(
                    "  {}: {:?}, dims={:?}",
                    tensor.name, tensor.quant_type, tensor.dims
                );
            }
        }

        // Print weight dimension analysis for key tensors
        println!("\n=== Weight Dimension Analysis ===");
        let key_patterns = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output.weight",
        ];

        for pattern in &key_patterns {
            if let Some(tensor) =
                self.gguf.tensors.iter().find(|t| t.name == *pattern)
            {
                println!(
                    "  {}: dims={:?} (format: [dim0, dim1, ...])",
                    pattern, tensor.dims
                );
                if tensor.dims.len() == 2 {
                    println!(
                        "    -> Assuming [out_features, in_features] = [{}, {}]",
                        tensor.dims[0], tensor.dims[1]
                    );
                }
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
