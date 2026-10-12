// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Inference runtime: connect model layer with operators

mod kv_cache;

pub use kv_cache::{KVCache, KVCacheBase, KVCacheFormat};

use crate::model::config::ModelConfig;
use crate::model::llama::LlamaModel;
use crate::model::tensor::DataType;
use crate::model::weights::WeightStore;
use crate::ops::sampling::{sample, SamplingStrategy};

/// Inference context
pub struct InferenceContext {
    pub config: ModelConfig,
    pub model: LlamaModel,
    pub device: Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl InferenceContext {
    pub fn from_gguf(path: &str, device: Device) -> Result<Self, String> {
        let store = WeightStore::from_gguf(path).map_err(|e| e.to_string())?;
        let gguf =
            crate::model::GgufFile::open(path).map_err(|e| e.to_string())?;
        let config = ModelConfig::from_gguf(&gguf);
        let model = LlamaModel::from_weights(&store, config.clone());

        Ok(InferenceContext { config, model, device })
    }

    /// Get embedding for token
    pub fn get_embedding(&self, token_id: u32) -> Option<Vec<f32>> {
        let embed = self.model.tok_embeddings.as_ref()?;
        let hidden_size = self.config.hidden_size;
        let dtype = embed.dtype;

        let bytes = embed.bytes();
        let offset = (token_id as usize) * hidden_size * dtype_size(dtype);

        if offset + hidden_size * dtype_size(dtype) > bytes.len() {
            return None;
        }

        let data = &bytes[offset..offset + hidden_size * dtype_size(dtype)];
        Some(dequant_to_f32(data, dtype, hidden_size))
    }

    /// Get layer count
    pub fn layer_count(&self) -> usize {
        self.model.layers.len()
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.model.param_count()
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model.tok_embeddings.is_some() && !self.model.layers.is_empty()
    }

    /// Generate tokens from prompt
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        strategy: &SamplingStrategy,
    ) -> Vec<u32> {
        let mut kv_cache = KVCache::new(
            &self.config,
            prompt_tokens.len() + max_tokens,
            KVCacheFormat::F32,
        );
        let mut generated = Vec::with_capacity(max_tokens);

        // Prefill: process prompt
        if let Some(logits) = self.model.forward(prompt_tokens, &mut kv_cache) {
            let first_token = sample(&logits, strategy) as u32;
            generated.push(first_token);
        }

        // Decode: generate remaining tokens
        for _ in 1..max_tokens {
            let eos = self.config.eos_token_id;
            if generated.last() == Some(&eos) {
                break;
            }

            let last_token =
                generated.last().copied().unwrap_or(self.config.bos_token_id);
            if let Some(logits) =
                self.model.forward(&[last_token], &mut kv_cache)
            {
                let next_token = sample(&logits, strategy) as u32;
                generated.push(next_token);
            } else {
                break;
            }
        }

        generated
    }
}

fn dtype_size(dtype: DataType) -> usize {
    match dtype {
        DataType::F32 => 4,
        DataType::F16 => 2,
        DataType::BF16 => 2,
        _ => 1, // Quantized types need special handling
    }
}

pub fn dequant_to_f32(data: &[u8], dtype: DataType, count: usize) -> Vec<f32> {
    match dtype {
        DataType::F32 => {
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 4] = data[i * 4..i * 4 + 4].try_into().unwrap();
                result.push(f32::from_le_bytes(bytes));
            }
            result
        }
        DataType::F16 => {
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 2] = data[i * 2..i * 2 + 2].try_into().unwrap();
                let f16 = half::f16::from_le_bytes(bytes);
                result.push(f16.to_f32());
            }
            result
        }
        DataType::BF16 => {
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 2] = data[i * 2..i * 2 + 2].try_into().unwrap();
                let bf16 = half::bf16::from_le_bytes(bytes);
                result.push(bf16.to_f32());
            }
            result
        }
        _ => {
            // For quantized types, return zeros (needs proper dequant)
            vec![0.0; count]
        }
    }
}
