// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Forward executor for CPU inference

use crate::backend::arm::{
    flash_attn::flash_attn_f32, gemv, quant, rms_norm::rms_norm_f32,
    rope::rope_f32,
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
        let _hidden_size = config.hidden_size;

        eprintln!(
            "\n=== Forward token {} at position {} ===",
            token_id, self.seq_pos
        );

        // 1. Embedding lookup
        let mut hidden = self.embed_token(token_id)?;

        if false {
            // Always print embedding stats for debugging
            let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = hidden.iter().sum();
            eprintln!(
                "[Embedding] token={} min={:.4} max={:.4} sum={:.4}",
                token_id, min, max, sum
            );
        }

        // 2. Process each layer
        for layer_idx in 0..config.n_layers {
            hidden = self.forward_layer(layer_idx, &hidden)?;
        }

        // 3. Final norm
        hidden = self.final_norm(&hidden)?;

        // 4. Output projection
        let logits = self.output_projection(&hidden)?;

        if false {
            let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[Logits] seq_pos={} min={:.4} max={:.4} len={}",
                self.seq_pos,
                min,
                max,
                logits.len()
            );

            // Find top-5 tokens
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!(
                "[Top-5 tokens] seq_pos={}: {:?}",
                self.seq_pos,
                &indexed[..5]
            );
        }

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
                // F16 embedding: shape is [hidden_size, vocab_size]
                // Need to read by columns (stride access)
                let n_vocab = emb_data.len() / (hidden_size * 2);
                eprintln!(
                    "[Embed] n_vocab={} hidden_size={}",
                    n_vocab, hidden_size
                );
                let mut embedding = vec![0.0f32; hidden_size];
                for h in 0..hidden_size {
                    let offset = h * n_vocab * 2 + (token_id as usize) * 2;
                    if offset + 2 > emb_data.len() {
                        return Err(format!(
                            "Token {} out of range at h={}",
                            token_id, h
                        ));
                    }
                    let bytes = &emb_data[offset..offset + 2];
                    embedding[h] =
                        half::f16::from_le_bytes(bytes.try_into().unwrap())
                            .to_f32();
                }
                Ok(embedding)
            }
            QuantType::Q4_K => {
                // Q4_K: 256 elements per block, 144 bytes per block
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let block_size = std::mem::size_of::<BlockQ4_K>();
                let row_offset = (token_id as usize) * nb * block_size;

                if row_offset + nb * block_size > emb_data.len() {
                    return Err(format!("Token {} out of range", token_id));
                }

                let mut embedding = vec![0.0f32; hidden_size];
                let block_data = &emb_data[row_offset..];

                // Dequantize each block
                crate::backend::arm::dequant::dequant_q4_k(
                    block_data,
                    &mut embedding,
                    nb * QK_K,
                );

                Ok(embedding)
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
                    let _dequant = crate::backend::arm::dequant::dequant_q6_k(
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

        // Debug: print input stats
        if self.seq_pos == 0 {
            let min = input.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[L{}-input] min={:.4} max={:.4}", layer_idx, min, max);
        }

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

        if self.seq_pos == 0
            && (layer_idx == 0 || layer_idx == 7 || layer_idx == 8)
        {
            let min = q.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = q.iter().sum();

            // Check the weight tensor type
            let q_weight_name = format!("blk.{}.attn_q.weight", layer_idx);
            let tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == q_weight_name);
            let quant_type = tensor
                .map(|t| format!("{:?}", t.quant_type))
                .unwrap_or("unknown".to_string());

            eprintln!(
                "[L{}-Q] min={:.4} max={:.4} sum={:.4} quant_type={}",
                layer_idx, min, max, sum, quant_type
            );
        }

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

        if self.seq_pos == 0
            && (layer_idx == 0 || layer_idx == 7 || layer_idx == 8)
        {
            let v_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let v_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let v_sum: f32 = v.iter().sum();
            eprintln!(
                "[L{}-V] min={:.4} max={:.4} sum={:.4}",
                layer_idx, v_min, v_max, v_sum
            );
        }

        // 3. Q/K norm (Qwen3 has Q norm)
        let q_normed = self.rms_norm_per_head(
            &q,
            &format!("blk.{}.attn_q_norm.weight", layer_idx),
            n_heads,
            head_dim,
        )?;
        let k_normed = self.rms_norm_per_head(
            &k,
            &format!("blk.{}.attn_k_norm.weight", layer_idx),
            n_kv_heads,
            head_dim,
        )?;

        if layer_idx == 0 && self.seq_pos == 0 {
            let q_min = q.iter().cloned().fold(f32::INFINITY, f32::min);
            let q_max = q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let qn_min = q_normed.iter().cloned().fold(f32::INFINITY, f32::min);
            let qn_max =
                q_normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[L0-Q_norm] before: [{:.4}, {:.4}] after: [{:.4}, {:.4}]",
                q_min, q_max, qn_min, qn_max
            );
        }

        // 4. Apply RoPE to Q and K
        let freq_base = config.rope_freq_base;
        let mut q_rope = q_normed.clone();
        let mut k_rope = k_normed.clone();
        for h in 0..n_heads {
            rope_f32(
                &q_normed[h * head_dim..(h + 1) * head_dim],
                &mut q_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                freq_base,
            );
        }
        for h in 0..n_kv_heads {
            rope_f32(
                &k_normed[h * head_dim..(h + 1) * head_dim],
                &mut k_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                freq_base,
            );
        }

        if layer_idx == 0 && self.seq_pos == 0 {
            let qr_min = q_rope.iter().cloned().fold(f32::INFINITY, f32::min);
            let qr_max =
                q_rope.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let kr_min = k_rope.iter().cloned().fold(f32::INFINITY, f32::min);
            let kr_max =
                k_rope.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[L0-RoPE] q_rope: [{:.4}, {:.4}] k_rope: [{:.4}, {:.4}] freq_base={}",
                qr_min, qr_max, kr_min, kr_max, freq_base
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

        if layer_idx == 0 && self.seq_pos <= 2 {
            eprintln!(
                "[L{}-KV] pos={} wrote {} floats to KV cache",
                layer_idx, self.seq_pos, head_dim
            );
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

            if layer_idx == 0
                && h == 0
                && (self.seq_pos <= 2 || self.seq_pos == 37)
            {
                eprintln!(
                    "[L{}-Attn] h={} n_kv={} seq_pos={} k_len={} v_len={}",
                    layer_idx,
                    h,
                    n_kv_heads,
                    self.seq_pos,
                    k_slice.len(),
                    v_slice.len()
                );
            }

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

        if self.seq_pos == 0
            && (layer_idx == 0 || layer_idx == 7 || layer_idx == 8)
        {
            let ao_min =
                attn_output.iter().cloned().fold(f32::INFINITY, f32::min);
            let ao_max =
                attn_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ao_sum: f32 = attn_output.iter().sum();
            eprintln!(
                "[L{}-attn_before_proj] min={:.4} max={:.4} sum={:.4}",
                layer_idx, ao_min, ao_max, ao_sum
            );
        }

        // 7. Output projection
        let attn_out = self.linear(
            &attn_output,
            &format!("blk.{}.attn_output.weight", layer_idx),
            hidden_size,
        )?;

        if self.seq_pos == 0 {
            let min = attn_out.iter().cloned().fold(f32::INFINITY, f32::min);
            let max =
                attn_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[L{}-attn_out] min={:.4} max={:.4}",
                layer_idx, min, max
            );
        }

        // Residual connection
        if self.seq_pos == 0 && layer_idx == 2 {
            eprintln!(
                "[L2-residual_debug] input_first5={:?} attn_out_first5={:?}",
                &input[..5.min(input.len())],
                &attn_out[..5.min(attn_out.len())]
            );
        }

        let hidden: Vec<f32> =
            input.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        if self.seq_pos == 0 && layer_idx == 2 {
            eprintln!(
                "[L2-residual_debug] hidden_first5={:?}",
                &hidden[..5.min(hidden.len())]
            );
        }

        // === FFN Block ===
        // 1. RMS Norm before FFN
        let normed_ffn = self.rms_norm(
            &hidden,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            hidden_size,
        )?;

        if self.seq_pos == 0
            && (layer_idx == 0 || layer_idx == 2 || layer_idx == 7)
        {
            let hidden_min =
                hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let hidden_max =
                hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let normed_min =
                normed_ffn.iter().cloned().fold(f32::INFINITY, f32::min);
            let normed_max =
                normed_ffn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[L{}-FFN-hidden_before_norm] min={:.4} max={:.4}",
                layer_idx, hidden_min, hidden_max
            );
            eprintln!(
                "[L{}-FFN-normed] min={:.4} max={:.4}",
                layer_idx, normed_min, normed_max
            );
        }

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

        if self.seq_pos == 0
            && (layer_idx == 0 || layer_idx == 2 || layer_idx == 7)
        {
            let gate_min = gate.iter().cloned().fold(f32::INFINITY, f32::min);
            let gate_max =
                gate.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let up_min = up.iter().cloned().fold(f32::INFINITY, f32::min);
            let up_max = up.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let gate_weight_name = format!("blk.{}.ffn_gate.weight", layer_idx);
            let up_weight_name = format!("blk.{}.ffn_up.weight", layer_idx);
            let gate_tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == gate_weight_name);
            let up_tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == up_weight_name);
            eprintln!(
                "[L{}-FFN-gate] min={:.4} max={:.4} quant={:?}",
                layer_idx,
                gate_min,
                gate_max,
                gate_tensor.map(|t| t.quant_type)
            );
            eprintln!(
                "[L{}-FFN-up] min={:.4} max={:.4} quant={:?}",
                layer_idx,
                up_min,
                up_max,
                up_tensor.map(|t| t.quant_type)
            );
            eprintln!(
                "[L{}-FFN-gate_first5] {:?}",
                layer_idx,
                &gate[..5.min(gate.len())]
            );
            eprintln!(
                "[L{}-FFN-up_first5] {:?}",
                layer_idx,
                &up[..5.min(up.len())]
            );

            // Debug: check weight scales for Q5_K
            if layer_idx == 2 {
                if let Some(gate_data) =
                    self.model.get_tensor(&gate_weight_name)
                {
                    let gate_blocks: &[crate::types::BlockQ5_K] = unsafe {
                        std::slice::from_raw_parts(
                            gate_data.as_ptr()
                                as *const crate::types::BlockQ5_K,
                            gate_data.len() / 176,
                        )
                    };
                    let first_d_f32 =
                        crate::types::fp16_to_fp32(gate_blocks[0].d);
                    let first_dmin_f32 =
                        crate::types::fp16_to_fp32(gate_blocks[0].dmin);
                    eprintln!(
                        "[L2-FFN-gate_weight] nb_blocks={} first_d={:.4} first_dmin={:.4}",
                        gate_blocks.len(),
                        first_d_f32,
                        first_dmin_f32
                    );
                    // Check scale distribution
                    let scales: Vec<f32> = gate_blocks
                        .iter()
                        .map(|b| crate::types::fp16_to_fp32(b.d))
                        .collect();
                    let scale_min =
                        scales.iter().cloned().fold(f32::INFINITY, f32::min);
                    let scale_max = scales
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    eprintln!(
                        "[L2-FFN-gate_weight_scales] min={:.6} max={:.6}",
                        scale_min, scale_max
                    );
                }
            }

            // Debug: check the actual normed input
            if layer_idx == 2 {
                eprintln!(
                    "[L2-FFN-normed_sample10] {:?}",
                    &normed_ffn[..10.min(normed_ffn.len())]
                );

                // Check Q8_K quantization of this input
                let nb = (normed_ffn.len() + crate::types::QK_K - 1)
                    / crate::types::QK_K;
                let mut input_q8k: Vec<crate::types::BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    &normed_ffn,
                    &mut input_q8k,
                );
                eprintln!(
                    "[L2-FFN-input_Q8_K] first_block_d={:.4}",
                    input_q8k[0].d
                );

                // Check which blocks have large scales
                let scales: Vec<f32> = input_q8k.iter().map(|b| b.d).collect();
                let scale_min =
                    scales.iter().cloned().fold(f32::INFINITY, f32::min);
                let scale_max =
                    scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let large_scales: Vec<(usize, f32)> = scales
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.abs() > 0.5)
                    .map(|(i, s)| (i, *s))
                    .collect();
                eprintln!(
                    "[L2-FFN-input_Q8_K] scales: min={:.4} max={:.4} large_count={}",
                    scale_min,
                    scale_max,
                    large_scales.len()
                );
                if !large_scales.is_empty() {
                    eprintln!(
                        "[L2-FFN-input_Q8_K] large scales: {:?}",
                        &large_scales[..large_scales.len().min(5)]
                    );
                }
            }
        }

        // 3. SwiGLU activation
        let mut ffn_hidden = vec![0.0f32; config.intermediate_size];
        crate::backend::arm::activation::silu_f32(&gate, &mut ffn_hidden);
        for i in 0..config.intermediate_size {
            ffn_hidden[i] *= up[i];
        }

        if self.seq_pos == 0
            && (layer_idx == 0 || layer_idx == 2 || layer_idx == 7)
        {
            let hidden_min =
                ffn_hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let hidden_max =
                ffn_hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[L{}-FFN-hidden] min={:.4} max={:.4}",
                layer_idx, hidden_min, hidden_max
            );
        }

        // 4. Down projection
        let ffn_out = self.linear(
            &ffn_hidden,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            hidden_size,
        )?;

        if self.seq_pos == 0
            && (layer_idx == 0 || layer_idx == 7 || layer_idx == 8)
        {
            let ffn_min = ffn_out.iter().cloned().fold(f32::INFINITY, f32::min);
            let ffn_max =
                ffn_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[L{}-ffn_out] min={:.4} max={:.4}",
                layer_idx, ffn_min, ffn_max
            );
        }

        // Add residual
        if self.seq_pos == 0 && (layer_idx == 0 || layer_idx == 2) {
            let h_min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let h_max =
                hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let f_min = ffn_out.iter().cloned().fold(f32::INFINITY, f32::min);
            let f_max =
                ffn_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[L{}-residual_inputs] hidden=[{:.4}, {:.4}] ffn_out=[{:.4}, {:.4}]",
                layer_idx, h_min, h_max, f_min, f_max
            );
            eprintln!(
                "[L{}-residual_first5] hidden={:?} ffn_out={:?}",
                layer_idx,
                &hidden[..5.min(hidden.len())],
                &ffn_out[..5.min(ffn_out.len())]
            );
        }

        let output: Vec<f32> =
            hidden.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        if self.seq_pos == 0 {
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[L{}-output] min={:.4} max={:.4}", layer_idx, min, max);
            eprintln!(
                "[L{}-output_first5] {:?}",
                layer_idx,
                &output[..5.min(output.len())]
            );
        }

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

        // Debug: compute RMS value
        let eps = 1e-5f32;
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / size as f32 + eps).sqrt();

        if let Some(data) = weight_data {
            // For F32 weight
            if data.len() == size * 4 {
                let weight: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                // Debug: check weight values
                let w_min =
                    weight.iter().cloned().fold(f32::INFINITY, f32::min);
                let w_max =
                    weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if weight_name.contains("ffn_norm")
                    && (weight_name.contains("blk.0")
                        || weight_name.contains("blk.2"))
                    && self.seq_pos == 0
                {
                    eprintln!(
                        "[{}] rms={:.4} w=[{:.4}, {:.4}]",
                        weight_name, rms, w_min, w_max
                    );
                }

                // Use NEON optimized RMS norm
                rms_norm_f32(input, &weight, &mut output, 1e-5);
            } else {
                // Fallback: just normalize without weight
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
            // Check if weight is F32 format
            let weight: Vec<f32> = if data.len() == n_heads * head_dim * 4 {
                // Per-head weights
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            } else if data.len() == head_dim * 4 {
                // Shared weight across all heads
                let shared_weight: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                // Repeat for all heads
                shared_weight.repeat(n_heads)
            } else if data.len() == head_dim * 2 {
                // F16 format, shared weight
                let shared_weight: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect();
                shared_weight.repeat(n_heads)
            } else {
                // Unknown format, just return input
                eprintln!(
                    "[Warning] Unknown weight format for {}: len={}",
                    weight_name,
                    data.len()
                );
                output.copy_from_slice(input);
                return Ok(output);
            };

            for h in 0..n_heads {
                let in_slice = &input[h * head_dim..(h + 1) * head_dim];
                let w_slice = &weight[h * head_dim..(h + 1) * head_dim];
                let out_slice = &mut output[h * head_dim..(h + 1) * head_dim];
                rms_norm_f32(in_slice, w_slice, out_slice, 1e-5);
            }
        } else {
            // Fallback: no norm
            output.copy_from_slice(input);
        }

        Ok(output)
    }

    /// Linear layer (GEMV) with quantized weights
    /// GGUF stores weights as dims = [in_features, out_features] in row-major order
    /// Each row of output has blocks_per_row blocks, stored contiguously
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

        // Verify dimensions match expected format
        // GGUF format: dims = [in_features, out_features]
        if self.seq_pos == 0 && weight_name.contains("blk.0") {
            eprintln!(
                "[Linear:{}] dims={:?} in={} out={}",
                weight_name, tensor.dims, in_features, out_features
            );
        }

        match tensor.quant_type {
            QuantType::F16 => {
                // F16 weights stored row-major: W[out_features, in_features]
                // dims = [in_features, out_features], but memory is row-major
                // output[i] = sum_j input[j] * W[i, j]
                let weights_f32: Vec<f32> = weight_data
                    .chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect();

                // Row-major access: each output row is contiguous
                for i in 0..out_features {
                    let mut sum = 0.0f32;
                    for j in 0..in_features {
                        let w_idx = i * in_features + j;
                        if w_idx < weights_f32.len() {
                            sum += input[j] * weights_f32[w_idx];
                        }
                    }
                    output[i] = sum;
                }
            }
            QuantType::Q6_K => {
                // Q6_K: 256 elements per block, row-major storage
                let nb = (in_features + QK_K - 1) / QK_K;

                // Quantize input to Q8_K
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                // Row-major: each output row has nb blocks
                let blocks_per_row = nb;
                let weights: &[BlockQ6_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const BlockQ6_K,
                        out_features * blocks_per_row,
                    )
                };

                for i in 0..out_features {
                    let row_blocks =
                        &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q6_k_q8_k(
                        row_blocks, &input_q8k,
                    );
                }
            }
            QuantType::Q5_K => {
                // Q5_K: 256 elements per block, row-major storage
                let nb = (in_features + QK_K - 1) / QK_K;

                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                // Row-major: each output row has nb blocks
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
                // Q4_K: 256 elements per block, row-major storage
                let nb = (in_features + QK_K - 1) / QK_K;

                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                // Row-major: each output row has nb blocks
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
            QuantType::IQ4_XS => {
                // IQ4_XS: 256 elements per block, uses non-linear quantization
                let nb = (in_features + QK_K - 1) / QK_K;

                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                let blocks_per_row = nb;

                let weights: &[BlockIQ4XS] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const BlockIQ4XS,
                        out_features * blocks_per_row,
                    )
                };

                for i in 0..out_features {
                    let row_blocks =
                        &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                    output[i] =
                        crate::backend::arm::vec_dot::vec_dot_iq4_xs_q8_k(
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
        let hidden_size = hidden.len();

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

        // Detect weight format
        // Shape [hidden_size, vocab_size] -> need transposed GEMV
        // Shape [vocab_size, hidden_size] -> normal GEMV
        let is_transposed = output_tensor.dims.len() == 2
            && output_tensor.dims[0] as usize == hidden_size;

        eprintln!(
            "[Output] shape={:?} is_transposed={}",
            output_tensor.dims, is_transposed
        );

        match output_tensor.quant_type {
            QuantType::Q6_K => {
                if is_transposed {
                    // Shape is [hidden_size, vocab_size]
                    // For quantized transposed GEMV, need to dequantize the whole weight
                    // This is slow but correct
                    let total_blocks = (output_tensor.dims[0] as usize
                        * output_tensor.dims[1] as usize
                        + QK_K
                        - 1)
                        / QK_K;
                    let block_size = std::mem::size_of::<BlockQ6_K>();

                    // Dequantize all weights
                    let total_elems = output_tensor.dims[0] as usize
                        * output_tensor.dims[1] as usize;
                    let mut weights_f32 = vec![0.0f32; total_elems];

                    for block_idx in 0..total_blocks {
                        let start_elem = block_idx * QK_K;
                        let offset = block_idx * block_size;
                        let remaining =
                            total_elems.saturating_sub(start_elem).min(QK_K);
                        if offset + block_size <= output_data.len()
                            && remaining > 0
                        {
                            let block_bytes =
                                &output_data[offset..offset + block_size];
                            let _dequant =
                                crate::backend::arm::dequant::dequant_q6_k(
                                    block_bytes,
                                    &mut weights_f32[start_elem..],
                                    remaining,
                                );
                        }
                    }

                    // Compute logits
                    let vocab_size_usize = vocab_size;
                    for v in 0..vocab_size_usize {
                        let mut sum = 0.0f32;
                        for h in 0..hidden_size {
                            sum += hidden[h]
                                * weights_f32[h * vocab_size_usize + v];
                        }
                        logits[v] = sum;
                    }
                } else {
                    // Normal GEMV: shape is [vocab_size, hidden_size]
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
            }
            QuantType::Q4_K => {
                if is_transposed {
                    // Shape is [hidden_size, vocab_size]
                    // Need dequantize and compute transposed GEMV
                    let total_blocks = (output_tensor.dims[0] as usize
                        * output_tensor.dims[1] as usize
                        + QK_K
                        - 1)
                        / QK_K;
                    let block_size = std::mem::size_of::<BlockQ4_K>();

                    let total_elems = output_tensor.dims[0] as usize
                        * output_tensor.dims[1] as usize;
                    let mut weights_f32 = vec![0.0f32; total_elems];

                    for block_idx in 0..total_blocks {
                        let start_elem = block_idx * QK_K;
                        let offset = block_idx * block_size;
                        let remaining =
                            total_elems.saturating_sub(start_elem).min(QK_K);
                        if offset + block_size <= output_data.len()
                            && remaining > 0
                        {
                            crate::backend::arm::dequant::dequant_q4_k(
                                &output_data[offset..offset + block_size],
                                &mut weights_f32[start_elem..],
                                remaining,
                            );
                        }
                    }

                    // Compute logits: hidden @ W = logits
                    // W[h, v] = weights_f32[h * vocab_size + v]
                    let vocab_size_usize = vocab_size;
                    for v in 0..vocab_size_usize {
                        let mut sum = 0.0f32;
                        for h in 0..hidden_size {
                            sum += hidden[h]
                                * weights_f32[h * vocab_size_usize + v];
                        }
                        logits[v] = sum;
                    }
                } else {
                    // Normal GEMV: shape is [vocab_size, hidden_size]
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

        // 1. Prefill: process all prompt tokens except the last one
        // The last token will be processed in decode phase
        println!("Prefilling {} tokens...", tokens.len() - 1);
        for (i, &token) in tokens[..tokens.len() - 1].iter().enumerate() {
            match self.forward(token) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Prefill error at token {}: {}", i, e);
                    return tokens;
                }
            }
        }
        println!("Prefill complete. seq_pos={}", self.seq_pos);

        // 2. Generate new tokens (including processing the last prompt token)
        for i in 0..max_new_tokens {
            // Forward pass for last token
            match self.forward(tokens[tokens.len() - 1]) {
                Ok(logits) => {
                    // Debug: check logits
                    if false {
                        let max_logit = logits
                            .iter()
                            .cloned()
                            .fold(f32::NEG_INFINITY, f32::max);
                        let min_logit = logits
                            .iter()
                            .cloned()
                            .fold(f32::INFINITY, f32::min);
                        println!(
                            "Decode {} logits: [{:.4}, {:.4}]",
                            i, min_logit, max_logit
                        );

                        // Find top-5 tokens
                        let mut indexed: Vec<(usize, f32)> =
                            logits.iter().cloned().enumerate().collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        println!("  Top-5: {:?}", &indexed[..5]);
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
