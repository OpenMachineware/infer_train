// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Forward executor for CPU inference

use crate::backend::arm::{
    activation::silu_f32_inplace, flash_attn::flash_attn_f32, linear, quant,
    rms_norm::rms_norm_f32, rope::rope_f32_inplace,
};
use crate::inference::SimpleModel;
use crate::types::*;

/// Quantized input storage with pre-allocated buffer
pub struct QuantizedInput {
    /// Q8_0 format blocks (for Q4_0, Q5_0, Q8_0 weights)
    pub q8_0: Vec<BlockQ8_0>,
    /// Q8_K format blocks (for Q4_K, Q5_K, Q6_K, Q8_K, IQ4_XS weights)
    pub q8_k: Vec<BlockQ8_K>,
    /// Current active format
    pub active: QuantFormat,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantFormat {
    None,
    Q8_0,
    Q8_K,
}

impl QuantizedInput {
    pub fn new() -> Self {
        Self { q8_0: Vec::new(), q8_k: Vec::new(), active: QuantFormat::None }
    }

    pub fn resize(&mut self, in_features: usize, format: QuantFormat) {
        match format {
            QuantFormat::Q8_0 => {
                let nb = (in_features + QK8_0 - 1) / QK8_0;
                if self.q8_0.len() != nb {
                    self.q8_0.resize(nb, BlockQ8_0 { d: 0, qs: [0; QK8_0] });
                }
            }
            QuantFormat::Q8_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                if self.q8_k.len() != nb {
                    self.q8_k.resize(nb, unsafe { std::mem::zeroed() });
                }
            }
            QuantFormat::None => {}
        }
        self.active = format;
    }
}

/// Forward executor for single-token generation
pub struct ForwardExecutor {
    pub model: SimpleModel,
    /// KV cache: [layer][seq_pos][head][dim] for key and value
    pub kv_cache_k: Vec<Vec<f32>>,
    pub kv_cache_v: Vec<Vec<f32>>,
    kv_capacity: usize,
    pub seq_pos: usize,
    quant_buf: QuantizedInput,
    arena: crate::memory::ComputeArena,
    hidden_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    pub n_threads: usize,
    pub n_threads_batch: usize,
}

impl ForwardExecutor {
    pub fn new(model: SimpleModel) -> Self {
        Self::new_with_threads(model, 0, 0) // Use optimal_threads() for decode
    }

    pub fn new_with_threads(
        model: SimpleModel,
        n_threads: usize,
        n_threads_batch: usize,
    ) -> Self {
        let config = model.config().clone();

        let work_size = crate::memory::calculate_work_size(
            config.hidden_size,
            config.intermediate_size,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim(),
            config.vocab_size,
            config.n_layers,
        );

        let kv_cache_k = vec![
            vec![
                0.0f32;
                config.n_kv_heads
                    * config.context_length
                    * config.head_dim()
            ];
            config.n_layers
        ];
        let kv_cache_v = vec![
            vec![
                0.0f32;
                config.n_kv_heads
                    * config.context_length
                    * config.head_dim()
            ];
            config.n_layers
        ];

        Self {
            model,
            kv_cache_k,
            kv_cache_v,
            kv_capacity: config.context_length,
            seq_pos: 0,
            quant_buf: QuantizedInput::new(),
            arena: crate::memory::ComputeArena::new(work_size),
            hidden_buf: vec![0.0f32; config.hidden_size],
            logits_buf: vec![0.0f32; config.vocab_size],
            n_threads,
            n_threads_batch,
        }
    }

    fn ensure_kv_capacity(&mut self) {
        let config = self.model.config();
        if self.seq_pos >= self.kv_capacity {
            let new_capacity =
                (self.kv_capacity * 2).min(config.context_length);
            for layer in 0..config.n_layers {
                self.kv_cache_k[layer].resize(
                    config.n_kv_heads * new_capacity * config.head_dim(),
                    0.0,
                );
                self.kv_cache_v[layer].resize(
                    config.n_kv_heads * new_capacity * config.head_dim(),
                    0.0,
                );
            }
            self.kv_capacity = new_capacity;
        }
    }

    pub fn reset(&mut self) {
        self.seq_pos = 0;
        for layer in &mut self.kv_cache_k {
            layer.fill(0.0);
        }
        for layer in &mut self.kv_cache_v {
            layer.fill(0.0);
        }
    }

    /// Forward pass returning reference to logits buffer
    pub fn forward(&mut self, token_id: u32) -> Result<&[f32], String> {
        self.arena.reset();
        self.ensure_kv_capacity();

        let n_layers = self.model.config().n_layers;

        // 1. Embedding lookup
        self.embed_token(token_id)?;

        // 2. Process each layer
        for layer_idx in 0..n_layers {
            self.forward_layer(layer_idx)?;
        }

        // 3. Final norm
        self.final_norm()?;

        // 4. Output projection
        self.output_projection()?;

        self.seq_pos += 1;
        Ok(&self.logits_buf)
    }

    fn embed_token(&mut self, token_id: u32) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

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
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;
                for h in 0..hidden_size {
                    self.hidden_buf[h] = f32::from_le_bytes(
                        emb_data[col_offset + h * 4..col_offset + h * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                }
            }
            QuantType::F16 => {
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;
                for h in 0..hidden_size {
                    self.hidden_buf[h] = half::f16::from_le_bytes(
                        emb_data[col_offset + h * 2..col_offset + h * 2 + 2]
                            .try_into()
                            .unwrap(),
                    )
                    .to_f32();
                }
            }
            QuantType::Q6_K => {
                let nb = hidden_size / 256;
                let block_size = std::mem::size_of::<BlockQ6_K>();
                let col_offset = (token_id as usize) * nb * block_size;
                for block_idx in 0..nb {
                    let h_base = block_idx * 256;
                    let block: &BlockQ6_K = unsafe {
                        &*(emb_data
                            .as_ptr()
                            .add(col_offset + block_idx * block_size)
                            as *const BlockQ6_K)
                    };
                    let block_vals =
                        crate::backend::arm::dequant::dequantize_q6_k_block(
                            block,
                        );
                    self.hidden_buf[h_base..h_base + 256]
                        .copy_from_slice(&block_vals);
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported embedding type: {:?}",
                    emb_tensor.quant_type
                ));
            }
        }
        Ok(())
    }

    fn forward_layer(&mut self, layer_idx: usize) -> Result<(), String> {
        self.arena.reset();

        let config = self.model.config().clone();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();

        // Attention norm
        let normed_attn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.rms_norm(
            &self.hidden_buf,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            normed_attn,
        )?;

        // Quantize input for projections
        let quant_type = self
            .get_weight_quant_type(&format!("blk.{}.attn_q.weight", layer_idx));
        let normed_len = normed_attn.len();
        let quant_format =
            quantize_input_to(normed_attn, quant_type, &mut self.quant_buf)?;
        let in_features = match quant_format {
            QuantFormat::Q8_0 => ((normed_len + QK8_0 - 1) / QK8_0) * QK8_0,
            QuantFormat::Q8_K => ((normed_len + QK_K - 1) / QK_K) * QK_K,
            QuantFormat::None => normed_len,
        };

        // QKV projections
        let q: &mut [f32] = self.arena.alloc_zeroed(n_heads * head_dim);
        let k: &mut [f32] = self.arena.alloc_zeroed(n_kv_heads * head_dim);
        let v: &mut [f32] = self.arena.alloc_zeroed(n_kv_heads * head_dim);

        self.linear_quantized(
            quant_format,
            in_features,
            &format!("blk.{}.attn_q.weight", layer_idx),
            q,
        )?;
        self.linear_quantized(
            quant_format,
            in_features,
            &format!("blk.{}.attn_k.weight", layer_idx),
            k,
        )?;
        self.linear_quantized(
            quant_format,
            in_features,
            &format!("blk.{}.attn_v.weight", layer_idx),
            v,
        )?;

        // Q/K norm
        let q_normed: &mut [f32] = self.arena.alloc_zeroed(n_heads * head_dim);
        let k_normed: &mut [f32] =
            self.arena.alloc_zeroed(n_kv_heads * head_dim);

        self.qk_norm(
            q,
            &format!("blk.{}.attn_q_norm.weight", layer_idx),
            n_heads,
            head_dim,
            q_normed,
        )?;
        self.qk_norm(
            k,
            &format!("blk.{}.attn_k_norm.weight", layer_idx),
            n_kv_heads,
            head_dim,
            k_normed,
        )?;

        // RoPE
        rope_f32_inplace(
            q_normed,
            self.seq_pos,
            head_dim,
            head_dim,
            config.rope_freq_base,
        );
        rope_f32_inplace(
            k_normed,
            self.seq_pos,
            head_dim,
            head_dim,
            config.rope_freq_base,
        );

        // KV cache update
        let kv_head_offset = self.seq_pos * head_dim;
        for h in 0..n_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = h * self.kv_capacity * head_dim + kv_head_offset;
            self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k_normed[src_offset..src_offset + head_dim]);
            self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }

        // Flash attention
        let attn_output: &mut [f32] =
            self.arena.alloc_zeroed(n_heads * head_dim);
        for h in 0..n_heads {
            let kv_head_idx = h * n_kv_heads / n_heads;
            let k_cache = &self.kv_cache_k[layer_idx]
                [kv_head_idx * self.kv_capacity * head_dim..];
            let v_cache = &self.kv_cache_v[layer_idx]
                [kv_head_idx * self.kv_capacity * head_dim..];

            flash_attn_f32(
                &q_normed[h * head_dim..(h + 1) * head_dim],
                &k_cache[..(self.seq_pos + 1) * head_dim],
                &v_cache[..(self.seq_pos + 1) * head_dim],
                None,
                &mut attn_output[h * head_dim..(h + 1) * head_dim],
                head_dim,
                self.seq_pos + 1,
                1.0 / (head_dim as f32).sqrt(),
            );
        }

        // Output projection
        let attn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear(
            attn_output,
            &format!("blk.{}.attn_output.weight", layer_idx),
            attn_out,
        )?;

        // Residual
        for i in 0..hidden_size {
            self.hidden_buf[i] += attn_out[i];
        }

        // FFN norm
        let normed_ffn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.rms_norm(
            &self.hidden_buf,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            normed_ffn,
        )?;

        // Quantize FFN input
        let ffn_normed_len = normed_ffn.len();
        let ffn_quant_format =
            quantize_input_to(normed_ffn, quant_type, &mut self.quant_buf)?;
        let ffn_in_features = match ffn_quant_format {
            QuantFormat::Q8_0 => ((ffn_normed_len + QK8_0 - 1) / QK8_0) * QK8_0,
            QuantFormat::Q8_K => ((ffn_normed_len + QK_K - 1) / QK_K) * QK_K,
            QuantFormat::None => ffn_normed_len,
        };

        // Gate/Up projections
        let gate: &mut [f32] =
            self.arena.alloc_zeroed(config.intermediate_size);
        let up: &mut [f32] = self.arena.alloc_zeroed(config.intermediate_size);

        self.linear_quantized(
            ffn_quant_format,
            ffn_in_features,
            &format!("blk.{}.ffn_gate.weight", layer_idx),
            gate,
        )?;
        self.linear_quantized(
            ffn_quant_format,
            ffn_in_features,
            &format!("blk.{}.ffn_up.weight", layer_idx),
            up,
        )?;

        // SwiGLU
        silu_f32_inplace(gate);
        for i in 0..config.intermediate_size {
            gate[i] *= up[i];
        }

        // Down projection
        let ffn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear(
            gate,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            ffn_out,
        )?;

        // Residual
        for i in 0..hidden_size {
            self.hidden_buf[i] += ffn_out[i];
        }

        Ok(())
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let size = input.len();
        let eps = self.model.config().rms_norm_eps;
        let weight_data = self.model.get_tensor(weight_name);

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / size as f32 + eps).sqrt();

        if let Some(data) = weight_data {
            if data.len() == size * 4 {
                let weight: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                rms_norm_f32(input, &weight, output, eps);
            } else {
                for (i, inp) in input.iter().enumerate() {
                    output[i] = inp / rms;
                }
            }
        } else {
            for (i, inp) in input.iter().enumerate() {
                output[i] = inp / rms;
            }
        }
        Ok(())
    }

    fn qk_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        n_heads: usize,
        head_dim: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        let eps = self.model.config().rms_norm_eps;
        let weight_data = self.model.get_tensor(weight_name);

        let shared_weight: Vec<f32> = if let Some(data) = weight_data {
            if data.len() == head_dim * 4 {
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            } else if data.len() == head_dim * 2 {
                data.chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect()
            } else {
                output.copy_from_slice(input);
                return Ok(());
            }
        } else {
            output.copy_from_slice(input);
            return Ok(());
        };

        for h in 0..n_heads {
            let head_start = h * head_dim;
            let head_input = &input[head_start..head_start + head_dim];
            let ss: f32 =
                head_input.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let scale = 1.0f32 / (ss + eps).sqrt();
            for d in 0..head_dim {
                output[head_start + d] =
                    head_input[d] * scale * shared_weight[d];
            }
        }
        Ok(())
    }

    fn linear(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
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

        let stride = tensor.strides.get(1).copied().unwrap_or(0) as usize;

        linear::linear_f32(
            input,
            weight_data,
            tensor.quant_type,
            stride,
            output,
            self.n_threads,
        )
    }

    fn linear_quantized(
        &self,
        quant_format: QuantFormat,
        _in_features: usize,
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
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

        let stride = tensor.strides.get(1).copied().unwrap_or(0) as usize;

        match quant_format {
            QuantFormat::Q8_0 => linear::linear_q8_0(
                &self.quant_buf.q8_0,
                self.quant_buf.q8_0.len() * QK8_0,
                weight_data,
                tensor.quant_type,
                stride,
                output,
                self.n_threads,
            ),
            QuantFormat::Q8_K => linear::linear_q8_k(
                &self.quant_buf.q8_k,
                self.quant_buf.q8_k.len() * QK_K,
                weight_data,
                tensor.quant_type,
                stride,
                output,
                self.n_threads,
            ),
            QuantFormat::None => Err("No quantized input available".into()),
        }
    }
}

/// Quantize input into the provided buffer
fn quantize_input_to(
    input: &[f32],
    quant_type: QuantType,
    quant_buf: &mut QuantizedInput,
) -> Result<QuantFormat, String> {
    match quant_type {
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            quant_buf.resize(input.len(), QuantFormat::Q8_0);
            quant::quantize_q8_0(input, &mut quant_buf.q8_0);
            Ok(QuantFormat::Q8_0)
        }
        QuantType::Q4_K
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::Q8_K
        | QuantType::IQ4_XS => {
            quant_buf.resize(input.len(), QuantFormat::Q8_K);
            quant::quantize_q8_k(input, &mut quant_buf.q8_k);
            Ok(QuantFormat::Q8_K)
        }
        _ => Err(format!("Unsupported quant type: {:?}", quant_type)),
    }
}

impl ForwardExecutor {
    fn get_weight_quant_type(&self, weight_name: &str) -> QuantType {
        self.model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == weight_name)
            .map(|t| t.quant_type)
            .unwrap_or(QuantType::F32)
    }

    fn final_norm(&mut self) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        let norm_weight = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == "output_norm.weight")
            .ok_or("output_norm.weight not found")?;

        let weight_data = self
            .model
            .get_tensor("output_norm.weight")
            .ok_or("output_norm weight data not found")?;

        let weight: Vec<f32> = match norm_weight.quant_type {
            QuantType::F32 => weight_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .take(hidden_size)
                .collect(),
            QuantType::F16 => weight_data
                .chunks_exact(2)
                .map(|chunk| {
                    half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f32()
                })
                .take(hidden_size)
                .collect(),
            _ => return Err("Unsupported norm quant type".into()),
        };

        rms_norm_f32(
            &self.hidden_buf.clone(),
            &weight,
            &mut self.hidden_buf,
            config.rms_norm_eps,
        );
        Ok(())
    }

    fn output_projection(&mut self) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = self.hidden_buf.len();

        let (output_tensor, output_data) = if let Some(tensor) =
            self.model.gguf.tensors.iter().find(|t| {
                t.name == "output.weight" || t.name == "lm_head.weight"
            }) {
            (
                tensor,
                self.model
                    .get_tensor(&tensor.name)
                    .ok_or("Output weight data not found")?,
            )
        } else {
            let tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .ok_or("No output or embedding weights found")?;
            (
                tensor,
                self.model
                    .get_tensor(&tensor.name)
                    .ok_or("Embedding weight data not found")?,
            )
        };

        self.logits_buf.fill(0.0);
        let stride = output_tensor.strides[1] as usize;

        // Quantize hidden_buf
        let quant_type = output_tensor.quant_type;
        if matches!(
            quant_type,
            QuantType::Q4_K | QuantType::Q5_K | QuantType::Q6_K
        ) {
            self.quant_buf.resize(hidden_size, QuantFormat::Q8_K);
            quant::quantize_q8_k(&self.hidden_buf, &mut self.quant_buf.q8_k);
            linear::linear_q8_k(
                &self.quant_buf.q8_k,
                hidden_size,
                output_data,
                quant_type,
                stride,
                &mut self.logits_buf,
                self.n_threads,
            )?;
        } else if matches!(quant_type, QuantType::Q4_0 | QuantType::Q8_0) {
            self.quant_buf.resize(hidden_size, QuantFormat::Q8_0);
            quant::quantize_q8_0(&self.hidden_buf, &mut self.quant_buf.q8_0);
            linear::linear_q8_0(
                &self.quant_buf.q8_0,
                hidden_size,
                output_data,
                quant_type,
                stride,
                &mut self.logits_buf,
                self.n_threads,
            )?;
        } else {
            linear::linear_f32(
                &self.hidden_buf,
                output_data,
                quant_type,
                stride,
                &mut self.logits_buf,
                self.n_threads,
            )?;
        }
        Ok(())
    }

    pub fn sample(&self, logits: &[f32], temperature: f32) -> u32 {
        if temperature == 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            let max_logit =
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits
                .iter()
                .map(|&l| ((l - max_logit) / temperature).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= sum;
            }

            use rand::Rng;
            let mut rng = rand::thread_rng();
            let r: f32 = rng.r#gen();
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= r {
                    return i as u32;
                }
            }
            (probs.len() - 1) as u32
        }
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        let mut tokens = prompt_tokens.to_vec();

        for (i, &token) in tokens.iter().enumerate() {
            if self.forward(token).is_err() {
                eprintln!("Prefill error at token {}", i);
                return tokens;
            }
        }

        for _ in 0..max_new_tokens {
            let next_token = {
                let logits = &self.logits_buf;
                self.sample(logits, temperature)
            };
            tokens.push(next_token);

            if self.forward(next_token).is_err() {
                break;
            }
        }

        tokens
    }
}
