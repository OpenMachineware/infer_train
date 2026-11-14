// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Forward executor for CPU inference

use crate::backend::arm::ArmCpuBackend;
use crate::inference::SimpleModel;
use crate::types::QuantType;

/// Forward executor for single-token generation
pub struct ForwardExecutor {
    model: SimpleModel,
    backend: ArmCpuBackend,
    /// KV cache (simplified: store as Vec<f32> for now)
    kv_cache: Vec<KVLayer>,
}

/// KV cache for a single layer
struct KVLayer {
    key: Vec<f32>,
    value: Vec<f32>,
}

impl ForwardExecutor {
    pub fn new(model: SimpleModel) -> Self {
        let config = model.config();
        let backend = ArmCpuBackend::new(0);

        // Initialize KV cache
        let mut kv_cache = Vec::with_capacity(config.n_layers);
        let kv_size =
            config.n_kv_heads * config.head_dim() * config.context_length;
        for _ in 0..config.n_layers {
            kv_cache.push(KVLayer {
                key: vec![0.0f32; kv_size],
                value: vec![0.0f32; kv_size],
            });
        }

        Self { model, backend, kv_cache }
    }

    /// Forward pass for a single token
    pub fn forward(&mut self, token_id: u32) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let _hidden_size = config.hidden_size;

        // 1. Embedding lookup
        let mut hidden = self.embed_token(token_id)?;

        // 2. Process each layer
        for layer_idx in 0..config.n_layers {
            hidden = self.forward_layer(layer_idx, &hidden)?;
        }

        // 3. Final norm
        hidden = self.final_norm(&hidden)?;

        // 4. Output projection
        let logits = self.output_projection(&hidden)?;

        Ok(logits)
    }

    /// Embedding lookup for a single token
    fn embed_token(&self, token_id: u32) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        // Find embedding weights
        let emb_data =
            self.model.get_embeddings().ok_or("No embedding weights")?;

        // Get embedding tensor info
        let emb_tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or("Embedding tensor not found")?;

        // Extract embedding for token_id based on quantization type
        match emb_tensor.quant_type {
            QuantType::F32 => {
                let offset = (token_id as usize) * hidden_size * 4;
                if offset + hidden_size * 4 > emb_data.len() {
                    return Err(format!("Token {} out of range", token_id));
                }
                let bytes = &emb_data[offset..offset + hidden_size * 4];
                let embedding: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Ok(embedding)
            }
            QuantType::F16 => {
                let offset = (token_id as usize) * hidden_size * 2;
                if offset + hidden_size * 2 > emb_data.len() {
                    return Err(format!("Token {} out of range", token_id));
                }
                let bytes = &emb_data[offset..offset + hidden_size * 2];
                let embedding: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect();
                Ok(embedding)
            }
            _ => {
                // For quantized embeddings, return zeros for now
                // TODO: Implement dequantization
                Ok(vec![0.0f32; hidden_size])
            }
        }
    }

    /// Forward pass for a single layer
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let _hidden_size = config.hidden_size;
        let _n_heads = config.n_heads;
        let _n_kv_heads = config.n_kv_heads;
        let _head_dim = config.head_dim();

        // === Attention Block ===
        // 1. RMS Norm before attention
        let normed_attn = self.rms_norm(
            input,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            config.hidden_size,
        )?;

        // 2. QKV projections (simplified - placeholder for now)
        // Real implementation would:
        // - Q = Linear(normed_attn, attn_q.weight)
        // - K = Linear(normed_attn, attn_k.weight)
        // - V = Linear(normed_attn, attn_v.weight)
        let _ = normed_attn; // Placeholder

        // 3. Q/K norm (if present - DeepSeek/Qwen3 style)
        // - Q = RMSNorm(Q, attn_q_norm.weight)
        // - K = RMSNorm(K, attn_k_norm.weight)

        // 4. Apply RoPE to Q and K

        // 5. Flash Attention

        // 6. Output projection

        // Residual connection
        let mut hidden: Vec<f32> = input.to_vec();

        // === FFN Block ===
        // 1. RMS Norm before FFN
        let normed_ffn = self.rms_norm(
            &hidden,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            config.hidden_size,
        )?;

        // 2. Gate and Up projections
        // Real implementation would:
        // - gate = Linear(normed_ffn, ffn_gate.weight)
        // - up = Linear(normed_ffn, ffn_up.weight)
        // - swiglu_out = SiLU(gate) * up
        // - down = Linear(swiglu_out, ffn_down.weight)

        // Residual connection (placeholder)
        // hidden = hidden + ffn_output
        let _ = normed_ffn; // Silence unused warning

        Ok(hidden)
    }

    /// RMS normalization
    fn rms_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        size: usize,
    ) -> Result<Vec<f32>, String> {
        // Find weight tensor
        let weight_data = self.model.get_tensor(weight_name);

        // Create output
        let mut output = vec![0.0f32; size];

        if let Some(data) = weight_data {
            // For F32 weight
            if data.len() == size * 4 {
                let weight: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                // Simple RMS norm implementation
                let eps = 1e-5f32;
                let sum_sq: f32 = input.iter().map(|x| x * x).sum();
                let rms = (sum_sq / size as f32 + eps).sqrt();

                for (i, (inp, w)) in input.iter().zip(weight.iter()).enumerate()
                {
                    output[i] = inp / rms * w;
                }
            }
        } else {
            // Fallback: just normalize without weight
            let eps = 1e-5f32;
            let sum_sq: f32 = input.iter().map(|x| x * x).sum();
            let rms = (sum_sq / size as f32 + eps).sqrt();

            for (i, inp) in input.iter().enumerate() {
                output[i] = inp / rms;
            }
        }

        Ok(output)
    }

    /// Final layer normalization
    fn final_norm(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        self.rms_norm(input, "output_norm.weight", config.hidden_size)
    }

    /// Output projection to logits
    fn output_projection(&self, _hidden: &[f32]) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let vocab_size = config.vocab_size;

        // Find output weights
        if let Some(_output_weight) = self.model.get_output_weight() {
            // TODO: Implement GEMV with output weights
        }

        // Return zeros as placeholder
        Ok(vec![0.0f32; vocab_size])
    }

    /// Sample next token from logits
    pub fn sample(&self, logits: &[f32], temperature: f32) -> u32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if temperature == 0.0 {
            // Greedy sampling
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            // Temperature sampling
            let max_logit =
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits
                .iter()
                .map(|&l| ((l - max_logit) / temperature).exp())
                .sum();

            let r = rng.r#gen::<f32>();
            let mut cumsum = 0.0;
            for (i, &logit) in logits.iter().enumerate() {
                cumsum += ((logit - max_logit) / temperature).exp() / exp_sum;
                if cumsum >= r {
                    return i as u32;
                }
            }
            (logits.len() - 1) as u32
        }
    }

    /// Generate tokens autoregressively
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        let mut tokens = prompt_tokens.to_vec();

        for _ in 0..max_new_tokens {
            // Forward pass for last token
            if let Ok(logits) = self.forward(tokens[tokens.len() - 1]) {
                let next_token = self.sample(&logits, temperature);
                tokens.push(next_token);
            } else {
                break;
            }
        }

        tokens
    }
}
