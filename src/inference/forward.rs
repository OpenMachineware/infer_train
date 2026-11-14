// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Forward executor for CPU inference

use crate::backend::arm::{
    dequant::dequant_q6_k, flash_attn::flash_attn_f32, gemv, quant,
    rms_norm::rms_norm_f32, rope::rope_f32,
};
use crate::inference::SimpleModel;
use crate::types::*;

/// Forward executor for single-token generation
pub struct ForwardExecutor {
    pub model: SimpleModel,
    /// KV cache: [layer][seq_pos][head][dim] for key and value
    pub kv_cache_k: Vec<Vec<f32>>,
    pub kv_cache_v: Vec<Vec<f32>>,
    /// Current sequence position
    pub seq_pos: usize,
}

impl ForwardExecutor {
    pub fn new(model: SimpleModel) -> Self {
        let config = model.config().clone();
        let n_layers = config.n_layers;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();
        let context_length = config.context_length;

        // Initialize KV cache
        let kv_size = n_kv_heads * head_dim * context_length;

        Self {
            model,
            kv_cache_k: vec![vec![0.0f32; kv_size]; n_layers],
            kv_cache_v: vec![vec![0.0f32; kv_size]; n_layers],
            seq_pos: 0,
        }
    }

    /// Forward pass for a single token
    pub fn forward(&mut self, token_id: u32) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

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

        // Increment sequence position
        self.seq_pos += 1;

        Ok(logits)
    }

    /// Embedding lookup for a single token
    fn embed_token(&self, token_id: u32) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        // Find embedding tensor info
        let emb_tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or("Embedding tensor not found")?;

        let emb_data =
            self.model.get_embeddings().ok_or("No embedding weights")?;

        match emb_tensor.quant_type {
            QuantType::F32 => {
                let offset = (token_id as usize) * hidden_size * 4;
                if offset + hidden_size * 4 > emb_data.len() {
                    return Err(format!("Token {} out of range", token_id));
                }
                let bytes = &emb_data[offset..offset + hidden_size * 4];
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect())
            }
            QuantType::F16 => {
                let offset = (token_id as usize) * hidden_size * 2;
                if offset + hidden_size * 2 > emb_data.len() {
                    return Err(format!("Token {} out of range", token_id));
                }
                let bytes = &emb_data[offset..offset + hidden_size * 2];
                Ok(bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect())
            }
            QuantType::Q6_K => {
                // Q6_K: 256 elements per block, 210 bytes per block
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let block_size = std::mem::size_of::<BlockQ6_K>();
                let row_offset = (token_id as usize) * nb * block_size;

                if row_offset + nb * block_size > emb_data.len() {
                    return Err(format!("Token {} out of range", token_id));
                }

                let mut embedding = vec![0.0f32; hidden_size];
                let block_data = &emb_data[row_offset..];

                // Dequantize each block
                for i in 0..nb {
                    let block_offset = i * block_size;
                    let block_bytes =
                        &block_data[block_offset..block_offset + block_size];
                    let dequant = dequant_q6_k(
                        block_bytes,
                        &mut embedding[i * QK_K..],
                        QK_K.min(hidden_size - i * QK_K),
                    );
                }

                Ok(embedding)
            }
            _ => Err(format!(
                "Unsupported embedding quantization: {:?}",
                emb_tensor.quant_type
            )),
        }
    }

    /// Forward pass for a single layer
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();

        // === Attention Block ===
        // 1. RMS Norm before attention
        let normed_attn = self.rms_norm(
            input,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            hidden_size,
        )?;

        // 2. QKV projections
        let q = self.linear(
            &normed_attn,
            &format!("blk.{}.attn_q.weight", layer_idx),
            n_heads * head_dim,
        )?;
        let k = self.linear(
            &normed_attn,
            &format!("blk.{}.attn_k.weight", layer_idx),
            n_kv_heads * head_dim,
        )?;
        let v = self.linear(
            &normed_attn,
            &format!("blk.{}.attn_v.weight", layer_idx),
            n_kv_heads * head_dim,
        )?;

        // 3. Q/K norm (Qwen3 has Q norm)
        let q = self.rms_norm_per_head(
            &q,
            &format!("blk.{}.attn_q_norm.weight", layer_idx),
            n_heads,
            head_dim,
        )?;
        let k = self.rms_norm_per_head(
            &k,
            &format!("blk.{}.attn_k_norm.weight", layer_idx),
            n_kv_heads,
            head_dim,
        )?;

        // 4. Apply RoPE to Q and K
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        for h in 0..n_heads {
            rope_f32(
                &q[h * head_dim..(h + 1) * head_dim],
                &mut q_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                10000.0,
            );
        }
        for h in 0..n_kv_heads {
            rope_f32(
                &k[h * head_dim..(h + 1) * head_dim],
                &mut k_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                10000.0,
            );
        }

        // 5. Update KV cache
        let kv_head_offset = self.seq_pos * head_dim;
        for h in 0..n_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset =
                h * config.context_length * head_dim + kv_head_offset;
            self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k_rope[src_offset..src_offset + head_dim]);
            self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }

        // 6. Flash Attention
        let mut attn_output = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv_head_idx = h * n_kv_heads / n_heads; // GQA mapping
            let k_cache = &self.kv_cache_k[layer_idx]
                [kv_head_idx * config.context_length * head_dim..];
            let v_cache = &self.kv_cache_v[layer_idx]
                [kv_head_idx * config.context_length * head_dim..];

            // Get K and V up to current position
            let k_slice = &k_cache[..(self.seq_pos + 1) * head_dim];
            let v_slice = &v_cache[..(self.seq_pos + 1) * head_dim];

            flash_attn_f32(
                &q_rope[h * head_dim..(h + 1) * head_dim],
                k_slice,
                v_slice,
                None,
                &mut attn_output[h * head_dim..(h + 1) * head_dim],
                head_dim,
                self.seq_pos + 1,
                1.0 / (head_dim as f32).sqrt(),
            );
        }

        // 7. Output projection
        let attn_out = self.linear(
            &attn_output,
            &format!("blk.{}.attn_output.weight", layer_idx),
            hidden_size,
        )?;

        // Residual connection
        let hidden: Vec<f32> =
            input.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // === FFN Block ===
        // 1. RMS Norm before FFN
        let normed_ffn = self.rms_norm(
            &hidden,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            hidden_size,
        )?;

        // 2. Gate and Up projections
        let gate = self.linear(
            &normed_ffn,
            &format!("blk.{}.ffn_gate.weight", layer_idx),
            config.intermediate_size,
        )?;
        let up = self.linear(
            &normed_ffn,
            &format!("blk.{}.ffn_up.weight", layer_idx),
            config.intermediate_size,
        )?;

        // 3. SwiGLU activation
        let mut ffn_hidden = vec![0.0f32; config.intermediate_size];
        crate::backend::arm::activation::silu_f32(&gate, &mut ffn_hidden);
        for i in 0..config.intermediate_size {
            ffn_hidden[i] *= up[i];
        }

        // 4. Down projection
        let ffn_out = self.linear(
            &ffn_hidden,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            hidden_size,
        )?;

        // Residual connection
        let output: Vec<f32> =
            hidden.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        Ok(output)
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

                // Use NEON optimized RMS norm
                rms_norm_f32(input, &weight, &mut output, 1e-5);
            } else {
                // Fallback: just normalize without weight
                let eps = 1e-5f32;
                let sum_sq: f32 = input.iter().map(|x| x * x).sum();
                let rms = (sum_sq / size as f32 + eps).sqrt();

                for (i, inp) in input.iter().enumerate() {
                    output[i] = inp / rms;
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

    /// RMS norm per head (for Q/K norm)
    fn rms_norm_per_head(
        &self,
        input: &[f32],
        weight_name: &str,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let weight_data = self.model.get_tensor(weight_name);

        let mut output = vec![0.0f32; n_heads * head_dim];

        if let Some(data) = weight_data {
            if data.len() == n_heads * head_dim * 4 {
                let weight: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                for h in 0..n_heads {
                    let in_slice = &input[h * head_dim..(h + 1) * head_dim];
                    let w_slice = &weight[h * head_dim..(h + 1) * head_dim];
                    let out_slice =
                        &mut output[h * head_dim..(h + 1) * head_dim];
                    rms_norm_f32(in_slice, w_slice, out_slice, 1e-5);
                }
            }
        } else {
            // Fallback: no norm
            output.copy_from_slice(input);
        }

        Ok(output)
    }

    /// Linear layer (GEMV) with quantized weights
    /// GGUF stores weights as [in_features, out_features] in dims,
    /// but the actual memory layout is [out_features][in_features]
    fn linear(
        &self,
        input: &[f32],
        weight_name: &str,
        out_features: usize,
    ) -> Result<Vec<f32>, String> {
        let in_features = input.len();

        // Find weight tensor
        let tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        // Create output
        let mut output = vec![0.0f32; out_features];

        match tensor.quant_type {
            QuantType::F16 => {
                // For F16, use transposed matmul
                let weights_f32: Vec<f32> = weight_data
                    .chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect();

                // output[i] = sum_j input[j] * weights[j * out_features + i]
                for i in 0..out_features {
                    let mut sum = 0.0f32;
                    for j in 0..in_features {
                        let w_idx = j * out_features + i;
                        if w_idx < weights_f32.len() {
                            sum += input[j] * weights_f32[w_idx];
                        }
                    }
                    output[i] = sum;
                }
            }
            QuantType::Q6_K => {
                // Q6_K: 256 elements per block
                let nb = (in_features + QK_K - 1) / QK_K;

                // Quantize input to Q8_K
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                // Number of blocks per row
                let blocks_per_row = nb;

                // Get weight blocks
                let weights: &[BlockQ6_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const BlockQ6_K,
                        out_features * blocks_per_row,
                    )
                };

                // Perform GEMV
                for i in 0..out_features {
                    let row_blocks =
                        &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q6_k_q8_k(
                        row_blocks, &input_q8k,
                    );
                }
            }
            QuantType::Q5_K => {
                // Q5_K: 256 elements per block
                let nb = (in_features + QK_K - 1) / QK_K;

                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                let blocks_per_row = nb;

                let weights: &[BlockQ5_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const BlockQ5_K,
                        out_features * blocks_per_row,
                    )
                };

                for i in 0..out_features {
                    let row_blocks =
                        &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q5_k_q8_k(
                        row_blocks, &input_q8k,
                    );
                }
            }
            QuantType::Q4_K => {
                // Q4_K: 256 elements per block
                let nb = (in_features + QK_K - 1) / QK_K;

                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                let blocks_per_row = nb;

                let weights: &[BlockQ4_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const BlockQ4_K,
                        out_features * blocks_per_row,
                    )
                };

                for i in 0..out_features {
                    let row_blocks =
                        &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q4_k_q8_k(
                        row_blocks, &input_q8k,
                    );
                }
            }
            QuantType::Q4_0 => {
                // Q4_0: 32 elements per block
                let nb = (in_features + QK8_0 - 1) / QK8_0;

                let mut input_q80: Vec<BlockQ8_0> =
                    vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
                crate::backend::arm::quant::quantize_q8_0(
                    input,
                    &mut input_q80,
                );

                let blocks_per_row = nb;

                let weights: &[BlockQ4_0] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const BlockQ4_0,
                        out_features * blocks_per_row,
                    )
                };

                for i in 0..out_features {
                    let row_blocks =
                        &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q4_0_q8_0(
                        row_blocks, &input_q80,
                    );
                }
            }
            QuantType::Q8_0 => {
                // Q8_0: 32 elements per block
                let nb = (in_features + QK8_0 - 1) / QK8_0;

                let mut input_q80: Vec<BlockQ8_0> =
                    vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
                crate::backend::arm::quant::quantize_q8_0(
                    input,
                    &mut input_q80,
                );

                let blocks_per_row = nb;

                let weights: &[BlockQ8_0] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const BlockQ8_0,
                        out_features * blocks_per_row,
                    )
                };

                for i in 0..out_features {
                    let row_blocks =
                        &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                    output[i] = crate::backend::arm::gemv::vec_dot_q8_0_single(
                        &row_blocks[0],
                        &input_q80[0],
                    );
                    // Sum all blocks
                    for (w, x) in row_blocks.iter().zip(input_q80.iter()) {
                        output[i] +=
                            crate::backend::arm::gemv::vec_dot_q8_0_single(
                                w, x,
                            );
                    }
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type for linear: {:?}",
                    tensor.quant_type
                ));
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
    fn output_projection(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let vocab_size = config.vocab_size;

        // Find output weights (try different names, including tied embeddings)
        let (output_tensor, output_data) = if let Some(tensor) =
            self.model.gguf.tensors.iter().find(|t| {
                t.name == "output.weight" || t.name == "lm_head.weight"
            }) {
            let data = self
                .model
                .get_tensor(&tensor.name)
                .ok_or("Output weight data not found")?;
            (tensor, data)
        } else {
            // Try tied embeddings
            let tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .ok_or("No output or embedding weights found")?;
            let data = self
                .model
                .get_tensor(&tensor.name)
                .ok_or("Embedding weight data not found")?;
            (tensor, data)
        };

        let in_features = hidden.len();
        let mut logits = vec![0.0f32; vocab_size];

        match output_tensor.quant_type {
            QuantType::Q6_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let weights: &[BlockQ6_K] = unsafe {
                    std::slice::from_raw_parts(
                        output_data.as_ptr() as *const BlockQ6_K,
                        vocab_size * nb,
                    )
                };

                let mut input_q8k = vec![unsafe { std::mem::zeroed() }; nb];
                quant::quantize_row_q8_k(hidden, &mut input_q8k);

                gemv::gemv_q6_k(
                    weights,
                    &input_q8k,
                    &mut logits,
                    vocab_size,
                    in_features,
                );
            }
            QuantType::Q4_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let weights: &[BlockQ4_K] = unsafe {
                    std::slice::from_raw_parts(
                        output_data.as_ptr() as *const BlockQ4_K,
                        vocab_size * nb,
                    )
                };

                let mut input_q8k = vec![unsafe { std::mem::zeroed() }; nb];
                quant::quantize_row_q8_k(hidden, &mut input_q8k);

                gemv::gemv_q4_k(
                    weights,
                    &input_q8k,
                    &mut logits,
                    vocab_size,
                    in_features,
                );
            }
            QuantType::Q5_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let weights: &[BlockQ5_K] = unsafe {
                    std::slice::from_raw_parts(
                        output_data.as_ptr() as *const BlockQ5_K,
                        vocab_size * nb,
                    )
                };

                let mut input_q8k = vec![unsafe { std::mem::zeroed() }; nb];
                quant::quantize_row_q8_k(hidden, &mut input_q8k);

                gemv::gemv_q5_k(
                    weights,
                    &input_q8k,
                    &mut logits,
                    vocab_size,
                    in_features,
                );
            }
            QuantType::F16 => {
                let n_weights = vocab_size * in_features;
                let weights_f32: Vec<f32> = output_data
                    .chunks_exact(2)
                    .take(n_weights)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect();

                for i in 0..vocab_size {
                    let w_row =
                        &weights_f32[i * in_features..(i + 1) * in_features];
                    logits[i] = crate::backend::arm::vec_dot::vec_dot_f32(
                        w_row, hidden,
                    );
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported output quantization: {:?}",
                    output_tensor.quant_type
                ));
            }
        }

        Ok(logits)
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

        println!("Generating {} new tokens...", max_new_tokens);

        for i in 0..max_new_tokens {
            // Forward pass for last token
            match self.forward(tokens[tokens.len() - 1]) {
                Ok(logits) => {
                    // Debug: check logits
                    if i == 0 {
                        let max_logit = logits
                            .iter()
                            .cloned()
                            .fold(f32::NEG_INFINITY, f32::max);
                        let min_logit = logits
                            .iter()
                            .cloned()
                            .fold(f32::INFINITY, f32::min);
                        println!(
                            "Logits range: [{:.4}, {:.4}]",
                            min_logit, max_logit
                        );
                    }
                    let next_token = self.sample(&logits, temperature);
                    println!("  Token {}: {}", i, next_token);
                    tokens.push(next_token);
                }
                Err(e) => {
                    eprintln!("Forward error at step {}: {}", i, e);
                    break;
                }
            }
        }

        tokens
    }
}
