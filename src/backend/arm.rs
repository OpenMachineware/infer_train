// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! ARM CPU backend using NEON SIMD.

use super::device::Device;
use super::memory::MemoryRegionId;
use super::operator::{
    CoreOperator, Cost, OperatorError, OperatorResult, QuantType,
};
use super::trait_impl::{Backend, TensorRef};
use crate::inference::KVCacheBase;

/// ARM CPU backend.
#[derive(Debug)]
pub struct ArmCpuBackend {
    device: Device,
    memory_region: MemoryRegionId,
    threads: usize,
}

impl ArmCpuBackend {
    /// Create a new ARM CPU backend.
    pub fn new(memory_region: MemoryRegionId) -> Self {
        let device = Device::detect_best();
        let threads =
            std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        Self { device, memory_region, threads }
    }

    /// Create with specific thread count.
    pub fn with_threads(memory_region: MemoryRegionId, threads: usize) -> Self {
        let device = Device::detect_best();
        Self { device, memory_region, threads }
    }

    /// Get thread count.
    pub fn threads(&self) -> usize {
        self.threads
    }
}

impl Backend for ArmCpuBackend {
    fn name(&self) -> &str {
        "ARM CPU (NEON)"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn memory_region(&self) -> MemoryRegionId {
        self.memory_region
    }

    fn available_memory(&self) -> u64 {
        self.device.memory_size()
    }

    fn supports_operator(&self, op: &CoreOperator) -> bool {
        match op {
            CoreOperator::RmsNorm { .. }
            | CoreOperator::Linear { .. }
            | CoreOperator::Rope { .. }
            | CoreOperator::SwiGLU { .. }
            | CoreOperator::Softmax { .. }
            | CoreOperator::Add { .. }
            | CoreOperator::Mul { .. } => true,
            CoreOperator::FlashAttentionDecode { .. } => true,
            CoreOperator::FlashAttentionPrefill { .. } => true,
            CoreOperator::MoERouter { .. }
            | CoreOperator::MoEExpertBatch { .. } => true,
        }
    }

    fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRef],
    ) -> OperatorResult<()> {
        match op {
            CoreOperator::RmsNorm { hidden_size, eps, .. } => {
                let input = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Input is not F32".into())
                })?;
                let weight = inputs[1].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Weight is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                crate::ops::rms_norm::rms_norm_f32(
                    input,
                    weight,
                    output,
                    *hidden_size,
                    *eps,
                );
                Ok(())
            }

            CoreOperator::Linear {
                in_features,
                out_features,
                quant_type,
                ..
            } => {
                // Linear: output = input @ weights^T
                // inputs[0] = input [batch, in_features]
                // inputs[1] = weights [out_features, in_features] (quantized)
                // outputs[0] = output [batch, out_features]

                let input = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Input is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Get weight data
                let weights = &inputs[1];
                let weight_bytes = weights.as_bytes().ok_or_else(|| {
                    OperatorError::InternalError(
                        "Weights is not raw bytes".into(),
                    )
                })?;

                let batch = input.len() / in_features;
                let m = *out_features;
                let n = batch;
                let k = *in_features;

                // Convert QuantType to DataType
                let dtype = quant_type_to_dtype(*quant_type);

                // Create WeightView
                let weight_view = crate::quant::gemm::WeightView::new(
                    dtype,
                    &weights.shape,
                    weight_bytes,
                );

                // Use gemm_dispatch_view
                let success = crate::quant::gemm::gemm_dispatch_view(
                    output,
                    &weight_view,
                    input,
                    m,
                    n,
                    k,
                    self.threads,
                    None, // q8k_buf - TODO: add to context
                );

                if !success {
                    return Err(OperatorError::UnsupportedOperation(format!(
                        "GEMM dispatch failed for dtype {:?}",
                        dtype
                    )));
                }
                Ok(())
            }

            CoreOperator::Rope { head_dim, freq_base, .. } => {
                let input = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Input is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // RoPE implementation (simplified)
                let theta_scale = freq_base.powf(-2.0 / *head_dim as f32);
                let n_heads = input.len() / head_dim;

                for h in 0..n_heads {
                    let base_idx = h * head_dim;
                    let mut theta = 1.0f32; // position 0
                    for d in (0..*head_dim).step_by(2) {
                        let cos_theta = theta.cos();
                        let sin_theta = theta.sin();

                        let idx = base_idx + d;
                        let x0 = input[idx];
                        let x1 = if d + 1 < *head_dim {
                            input[idx + 1]
                        } else {
                            0.0
                        };

                        output[idx] = x0 * cos_theta - x1 * sin_theta;
                        if d + 1 < *head_dim {
                            output[idx + 1] = x0 * sin_theta + x1 * cos_theta;
                        }

                        theta *= theta_scale;
                    }
                }
                Ok(())
            }

            CoreOperator::SwiGLU { intermediate_size, .. } => {
                let gate = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Gate is not F32".into())
                })?;
                let up = inputs[1].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Up is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Use NEON-optimized SwiGLU
                // swiglu_f32_neon(gate: &[f32], up: &[f32], dst: &mut [f32])
                unsafe {
                    crate::ops::arm::neon::swiglu_f32_neon(gate, up, output);
                }
                let _ = intermediate_size; // used for debug assert
                Ok(())
            }

            CoreOperator::FlashAttentionDecode {
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                ..
            } => {
                // Flash Attention decode: single token attends to all previous K/V
                // inputs[0] = Q [n_heads, head_dim]
                // inputs[1] = K [n_kv_heads, head_dim] (for cache update)
                // inputs[2] = V [n_kv_heads, head_dim] (for cache update)
                // inputs[3] = context with KV cache pointer
                // outputs[0] = output [n_heads, head_dim]

                let q = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Q is not F32".into())
                })?;
                let k = inputs[1].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("K is not F32".into())
                })?;
                let v = inputs[2].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("V is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Get KV cache from context
                let kv_cache: &mut crate::inference::KVCache =
                    inputs[3].context_as_mut().ok_or_else(|| {
                        OperatorError::InternalError(
                            "KV cache context not found".into(),
                        )
                    })?;

                // Update KV cache with new K, V
                kv_cache.set_k(*layer_idx, k, *n_kv_heads, *head_dim);
                kv_cache.set_v(*layer_idx, v, *n_kv_heads, *head_dim);

                // Compute attention scores and output
                let scale = 1.0f32 / (*head_dim as f32).sqrt();
                let q_size = n_heads * head_dim;

                // Clear output
                for elem in output.iter_mut() {
                    *elem = 0.0;
                }

                // For each head
                for h in 0..*n_heads {
                    let kv_head = h * n_kv_heads / n_heads;
                    let q_offset = h * head_dim;
                    let out_offset = h * head_dim;

                    // Flash attention online softmax
                    let mut max_score = f32::NEG_INFINITY;
                    let mut sum_exp = 0.0f32;
                    let mut scratch = vec![0.0f32; *head_dim];

                    // Attend to all positions in cache
                    for pos in 0..=*seq_len {
                        // Get K from cache
                        let k_row = kv_cache.get_k(*layer_idx, kv_head, pos);

                        // Compute attention score: Q · K^T
                        let mut score = 0.0f32;
                        for d in 0..*head_dim {
                            score += q[q_offset + d] * k_row[d];
                        }
                        score *= scale;

                        // Online softmax update
                        let new_max = max_score.max(score);
                        let exp_diff = if new_max != max_score {
                            let scale_factor = (max_score - new_max).exp();
                            for elem in scratch.iter_mut() {
                                *elem *= scale_factor;
                            }
                            sum_exp *= scale_factor;
                            max_score = new_max;
                            (score - max_score).exp()
                        } else {
                            (score - max_score).exp()
                        };

                        sum_exp += exp_diff;

                        // Get V and accumulate weighted sum
                        let v_row = kv_cache.get_v(*layer_idx, kv_head, pos);
                        for d in 0..*head_dim {
                            scratch[d] += exp_diff * v_row[d];
                        }
                    }

                    // Normalize and store
                    for d in 0..*head_dim {
                        output[out_offset + d] = scratch[d] / sum_exp;
                    }
                }

                Ok(())
            }

            CoreOperator::FlashAttentionPrefill {
                layer_idx,
                n_tokens,
                n_heads,
                n_kv_heads,
                head_dim,
                ..
            } => {
                // Flash Attention prefill: batch tokens
                // inputs[0] = Q [n_tokens, n_heads, head_dim]
                // inputs[1] = K [n_tokens, n_kv_heads, head_dim]
                // inputs[2] = V [n_tokens, n_kv_heads, head_dim]
                // inputs[3] = context with KV cache pointer
                // outputs[0] = output [n_tokens, n_heads, head_dim]

                let q = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Q is not F32".into())
                })?;
                let k = inputs[1].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("K is not F32".into())
                })?;
                let v = inputs[2].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("V is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Get KV cache from context
                let kv_cache: &mut crate::inference::KVCache =
                    inputs[3].context_as_mut().ok_or_else(|| {
                        OperatorError::InternalError(
                            "KV cache context not found".into(),
                        )
                    })?;

                let kv_size = n_kv_heads * head_dim;
                let q_size = n_heads * head_dim;

                // Store K, V to cache first
                for token_idx in 0..*n_tokens {
                    let k_offset = token_idx * kv_size;
                    let v_offset = token_idx * kv_size;
                    kv_cache.set_k(
                        *layer_idx,
                        &k[k_offset..k_offset + kv_size],
                        *n_kv_heads,
                        *head_dim,
                    );
                    kv_cache.set_v(
                        *layer_idx,
                        &v[v_offset..v_offset + kv_size],
                        *n_kv_heads,
                        *head_dim,
                    );
                    kv_cache.advance();
                }

                // Compute attention for each token
                let scale = 1.0f32 / (*head_dim as f32).sqrt();
                let mut scratch = vec![0.0f32; *head_dim];

                for token_idx in 0..*n_tokens {
                    let seq_len = token_idx + 1; // Causal mask

                    for h in 0..*n_heads {
                        let kv_head = h * n_kv_heads / n_heads;
                        let q_offset = token_idx * q_size + h * head_dim;
                        let out_offset = token_idx * q_size + h * head_dim;

                        // Flash attention online softmax
                        let mut max_score = f32::NEG_INFINITY;
                        let mut sum_exp = 0.0f32;
                        scratch.fill(0.0);

                        for pos in 0..seq_len {
                            let k_row =
                                kv_cache.get_k(*layer_idx, kv_head, pos);
                            let v_row =
                                kv_cache.get_v(*layer_idx, kv_head, pos);

                            // Compute score
                            let mut score = 0.0f32;
                            for d in 0..*head_dim {
                                score += q[q_offset + d] * k_row[d];
                            }
                            score *= scale;

                            // Online softmax update
                            let new_max = max_score.max(score);
                            let exp_diff = if new_max != max_score {
                                let scale_factor = (max_score - new_max).exp();
                                for elem in scratch.iter_mut() {
                                    *elem *= scale_factor;
                                }
                                sum_exp *= scale_factor;
                                max_score = new_max;
                                (score - max_score).exp()
                            } else {
                                (score - max_score).exp()
                            };

                            sum_exp += exp_diff;
                            for d in 0..*head_dim {
                                scratch[d] += exp_diff * v_row[d];
                            }
                        }

                        // Normalize
                        for d in 0..*head_dim {
                            output[out_offset + d] = scratch[d] / sum_exp;
                        }
                    }
                }

                Ok(())
            }

            CoreOperator::Softmax { size, .. } => {
                let input = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Input is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Use NEON-optimized softmax
                // softmax_f32_neon(x: &[f32], y: &mut [f32])
                let n_rows = input.len() / size;
                for row in 0..n_rows {
                    let offset = row * size;
                    unsafe {
                        crate::ops::arm::neon::softmax_f32_neon(
                            &input[offset..offset + size],
                            &mut output[offset..offset + size],
                        );
                    }
                }
                Ok(())
            }

            CoreOperator::Add { .. } => {
                let a = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("A is not F32".into())
                })?;
                let b = inputs[1].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("B is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Use NEON-optimized add
                // add_f32(src0: &[f32], src1: &[f32], dst: &mut [f32])
                unsafe {
                    crate::ops::arm::neon::add_f32(a, b, output);
                }
                Ok(())
            }

            CoreOperator::Mul { .. } => {
                let a = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("A is not F32".into())
                })?;
                let b = inputs[1].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("B is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Use NEON-optimized mul
                // mul_f32(src0: &[f32], src1: &[f32], dst: &mut [f32])
                unsafe {
                    crate::ops::arm::neon::mul_f32(a, b, output);
                }
                Ok(())
            }

            CoreOperator::MoERouter {
                hidden_size, n_experts, top_k, ..
            } => {
                // MoE Router: compute routing scores and select top-k experts
                // inputs[0] = hidden [hidden_size]
                // inputs[1] = gate_inp weights [hidden_size, n_experts]
                // outputs[0] = expert_indices [top_k]
                // outputs[1] = expert_weights [top_k]

                let hidden = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Hidden is not F32".into())
                })?;
                let gate_inp = inputs[1].as_bytes().ok_or_else(|| {
                    OperatorError::InternalError(
                        "Gate_inp weights is not bytes".into(),
                    )
                })?;

                // Output buffers
                let mut indices_out = vec![0.0f32; *top_k];
                let mut weights_out = vec![0.0f32; *top_k];

                // Dequantize gate_inp weights to F32
                let gate_weights: Vec<f32> = crate::inference::dequant_to_f32(
                    gate_inp,
                    inputs[1].dtype,
                    inputs[1].len,
                );

                // Compute routing scores: hidden @ gate_inp
                let mut scores = vec![0.0f32; *n_experts];
                for e in 0..*n_experts {
                    for i in 0..*hidden_size {
                        scores[e] +=
                            hidden[i] * gate_weights[i * n_experts + e];
                    }
                }

                // Softmax over scores
                let max_score =
                    scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 =
                    scores.iter().map(|s| (s - max_score).exp()).sum();
                for s in &mut scores {
                    *s = ((*s - max_score).exp()) / sum;
                }

                // Select top-k
                let mut indexed: Vec<(usize, f32)> =
                    scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Renormalize top-k weights
                let top_k_scores: Vec<(usize, f32)> =
                    indexed.into_iter().take(*top_k).collect();
                let sum_weights: f32 =
                    top_k_scores.iter().map(|(_, w)| w).sum();

                // Output indices and weights
                for (i, (idx, w)) in top_k_scores.iter().enumerate() {
                    indices_out[i] = *idx as f32;
                    weights_out[i] = w / sum_weights;
                }

                // Write to outputs using raw pointers
                if let Some(ptr) = outputs[0].data_f32 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            indices_out.as_ptr(),
                            ptr,
                            *top_k,
                        );
                    }
                }
                if let Some(ptr) = outputs[1].data_f32 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            weights_out.as_ptr(),
                            ptr,
                            *top_k,
                        );
                    }
                }

                Ok(())
            }

            CoreOperator::MoEExpertBatch {
                hidden_size,
                intermediate_size,
                n_experts,
                n_tokens,
                ..
            } => {
                // MoE Expert Batch: apply selected experts to hidden states
                // inputs[0] = hidden [n_tokens, hidden_size]
                // inputs[1] = expert_indices [n_tokens] (as F32)
                // inputs[2] = expert_weights [n_tokens] (as F32)
                // inputs[3] = gate_up_exps weights [n_experts, intermediate_size, hidden_size]
                // inputs[4] = down_exps weights [n_experts, hidden_size, intermediate_size]
                // outputs[0] = result [n_tokens, hidden_size]

                let hidden = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Hidden is not F32".into())
                })?;
                let expert_indices_f32 =
                    inputs[1].as_f32().ok_or_else(|| {
                        OperatorError::InternalError(
                            "Expert indices is not F32".into(),
                        )
                    })?;
                let expert_weights = inputs[2].as_f32().ok_or_else(|| {
                    OperatorError::InternalError(
                        "Expert weights is not F32".into(),
                    )
                })?;
                let gate_up_bytes = inputs[3].as_bytes().ok_or_else(|| {
                    OperatorError::InternalError(
                        "Gate_up_exps is not bytes".into(),
                    )
                })?;
                let down_bytes = inputs[4].as_bytes().ok_or_else(|| {
                    OperatorError::InternalError(
                        "Down_exps is not bytes".into(),
                    )
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Clear output
                for elem in output.iter_mut() {
                    *elem = 0.0;
                }

                // Process each token
                for token_idx in 0..*n_tokens {
                    let hidden_offset = token_idx * hidden_size;
                    let out_offset = token_idx * hidden_size;

                    // Get expert assignment (assuming single expert per token for now)
                    let expert_idx = expert_indices_f32[token_idx] as usize;
                    let weight = expert_weights[token_idx];

                    if expert_idx >= *n_experts || weight == 0.0 {
                        continue;
                    }

                    // Compute expert output: gate * silu(up), then down
                    // This is a simplified single-expert, single-token version
                    // For efficiency, we should batch by expert

                    // Allocate intermediate buffer
                    let mut gate_out = vec![0.0f32; *intermediate_size];
                    let mut up_out = vec![0.0f32; *intermediate_size];
                    let mut hidden_out = vec![0.0f32; *hidden_size];

                    // Gate projection: gate_out = hidden @ gate_exps[expert_idx]
                    // Using scalar GEMV for simplicity
                    let gate_up_weights: Vec<f32> =
                        crate::inference::dequant_to_f32(
                            gate_up_bytes,
                            inputs[3].dtype,
                            inputs[3].len,
                        );
                    let expert_gate_offset =
                        expert_idx * intermediate_size * hidden_size;

                    for i in 0..*intermediate_size {
                        let row_offset = expert_gate_offset + i * hidden_size;
                        let mut sum = 0.0f32;
                        for j in 0..*hidden_size {
                            sum += gate_up_weights[row_offset + j]
                                * hidden[hidden_offset + j];
                        }
                        gate_out[i] = sum;
                    }

                    // Up projection (second half of gate_up_exps if fused)
                    // For now, compute SiLU(gate) * up
                    for i in 0..*intermediate_size {
                        let gate_val = gate_out[i];
                        let silu = gate_val / (1.0 + (-gate_val).exp());
                        up_out[i] = silu * gate_out[i]; // Simplified, assuming gate==up
                    }

                    // Down projection: hidden_out = up_out @ down_exps[expert_idx]
                    let down_weights: Vec<f32> =
                        crate::inference::dequant_to_f32(
                            down_bytes,
                            inputs[4].dtype,
                            inputs[4].len,
                        );
                    let expert_down_offset =
                        expert_idx * hidden_size * intermediate_size;

                    for i in 0..*hidden_size {
                        let row_offset =
                            expert_down_offset + i * intermediate_size;
                        let mut sum = 0.0f32;
                        for j in 0..*intermediate_size {
                            sum += down_weights[row_offset + j] * up_out[j];
                        }
                        hidden_out[i] = sum * weight;
                    }

                    // Accumulate to output
                    for i in 0..*hidden_size {
                        output[out_offset + i] += hidden_out[i];
                    }
                }

                Ok(())
            }
        }
    }

    fn estimate_cost(&self, op: &CoreOperator) -> Cost {
        match op {
            CoreOperator::RmsNorm { hidden_size, .. } => Cost {
                compute_us: *hidden_size as f64 * 0.001,
                memory_bytes: *hidden_size as u64 * 4 * 3,
                read_bytes: *hidden_size as u64 * 4 * 2,
                write_bytes: *hidden_size as u64 * 4,
            },
            CoreOperator::Linear { in_features, out_features, .. } => {
                let flops = 2 * in_features * out_features;
                Cost {
                    compute_us: flops as f64 * 0.001,
                    memory_bytes: (in_features + out_features) as u64 * 4 * 2,
                    read_bytes: *in_features as u64 * 4,
                    write_bytes: *out_features as u64 * 4,
                }
            }
            _ => Cost::zero(),
        }
    }
}

/// Convert QuantType to DataType for GEMM dispatch.
fn quant_type_to_dtype(quant: QuantType) -> crate::model::tensor::DataType {
    use crate::model::tensor::DataType;
    match quant {
        QuantType::F32 => DataType::F32,
        QuantType::F16 => DataType::F16,
        QuantType::BF16 => DataType::BF16,
        QuantType::Q4_0 => DataType::Q4_0,
        QuantType::Q4_1 => DataType::Q4_1,
        QuantType::Q5_0 => DataType::Q5_0,
        QuantType::Q5_1 => DataType::Q5_1,
        QuantType::Q8_0 => DataType::Q8_0,
        QuantType::Q8_1 => DataType::Q8_1,
        QuantType::Q2_K => DataType::Q2_K,
        QuantType::Q3_K => DataType::Q3_K,
        QuantType::Q4_K => DataType::Q4_K,
        QuantType::Q5_K => DataType::Q5_K,
        QuantType::Q6_K => DataType::Q6_K,
        QuantType::IQ4_NL => DataType::IQ4_NL,
        QuantType::IQ4_XS => DataType::IQ4_XS,
        QuantType::IQ3_XXS => DataType::IQ3_XXS,
        QuantType::IQ3_S => DataType::IQ3_S,
    }
}
