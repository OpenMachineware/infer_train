// SPDX-License-Identifier: AGPL-3.0-or-later
// CPU work buffers for zero-allocation inference

/// Pre-allocated work buffers for CPU inference
pub struct CpuWorkBuffers {
    pub hidden: Vec<f32>,
    pub hidden_alt: Vec<f32>,
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub x_q8k: Vec<u8>,
    pub x_padded: Vec<f32>,

    // Additional buffers for zero-allocation
    pub q8k_input: Vec<u8>,
    pub qkv_buffer: Vec<f32>,
    pub q_buffer: Vec<f32>,
    pub k_buffer: Vec<f32>,
    pub v_buffer: Vec<f32>,
    pub rope_buffer: Vec<f32>,
    pub attn_scores: Vec<f32>,

    // Dequantized norm weights (flat array: each norm is hidden_size elements)
    pub norm_weights: Vec<f32>,
    pub norm_weight: Vec<f32>, // Final output norm

    // Config
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

impl CpuWorkBuffers {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_length: usize,
        value_length: usize,
        vocab_size: usize,
        qkv_out_size: usize,
        _max_seq_len: usize,
        n_layers: usize,
    ) -> Self {
        let q_size = n_heads * head_dim;
        // Use key_length and value_length for K/V buffer sizes (Qwen3.8-style)
        // Fall back to head_dim if key_length/value_length are 0
        let k_size = if key_length > 0 {
            n_kv_heads * key_length
        } else {
            n_kv_heads * head_dim
        };
        let v_size = if value_length > 0 {
            n_kv_heads * value_length
        } else {
            n_kv_heads * head_dim
        };
        let kv_size = k_size.max(v_size);

        // Attention output size: n_heads * v_head_dim
        let v_head_dim = if value_length > 0 { value_length } else { head_dim };
        let attn_out_size = n_heads * v_head_dim;

        Self {
            hidden: vec![0.0; hidden_size],
            hidden_alt: vec![0.0; hidden_size],
            normed: vec![0.0; hidden_size],
            q: vec![0.0; q_size],
            k: vec![0.0; kv_size],
            v: vec![0.0; kv_size],
            attn_out: vec![0.0; attn_out_size],
            ffn_gate: vec![0.0; intermediate_size],
            ffn_up: vec![0.0; intermediate_size],
            ffn_hidden: vec![0.0; intermediate_size],
            logits: vec![0.0; vocab_size],
            x_q8k: vec![0; 65536],
            x_padded: vec![0.0; 65536],
            q8k_input: vec![0; 65536],
            qkv_buffer: vec![0.0; qkv_out_size.max(q_size + 2 * kv_size)],
            q_buffer: vec![0.0; q_size],
            k_buffer: vec![0.0; kv_size],
            v_buffer: vec![0.0; kv_size],
            rope_buffer: vec![0.0; q_size.max(kv_size)],
            attn_scores: vec![0.0; 2048], // max_seq_len
            norm_weights: Vec::with_capacity(n_layers * 2 * hidden_size),
            norm_weight: vec![0.0; hidden_size],
            hidden_size,
            intermediate_size,
            vocab_size,
        }
    }

    pub fn init_norm_weights(&mut self, model: &crate::model::LlamaModel) {
        let hidden_size = self.hidden_size;
        for layer in model.layers.iter() {
            // Attention norm
            if let Some(ref norm) = layer.attention_norm {
                let total_elems = norm.elem_count();
                let data = crate::inference::dequant_to_f32(
                    norm.bytes(),
                    norm.dtype,
                    total_elems,
                );
                self.norm_weights.extend_from_slice(&data);
            } else {
                self.norm_weights.extend_from_slice(&vec![0.0; hidden_size]);
            }
            // FFN norm
            if let Some(ref norm) = layer.ffn_norm {
                let total_elems = norm.elem_count();
                let data = crate::inference::dequant_to_f32(
                    norm.bytes(),
                    norm.dtype,
                    total_elems,
                );
                self.norm_weights.extend_from_slice(&data);
            } else {
                self.norm_weights.extend_from_slice(&vec![0.0; hidden_size]);
            }
        }
        // Final output norm
        if let Some(ref norm) = model.norm {
            let total_elems = norm.elem_count();
            let data = crate::inference::dequant_to_f32(
                norm.bytes(),
                norm.dtype,
                total_elems,
            );
            self.norm_weight = data;
        }
    }
}
