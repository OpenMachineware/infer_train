// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! ARM CPU backend using NEON SIMD.

use super::device::Device;
use super::memory::MemoryRegionId;
use super::operator::{
    CoreOperator, Cost, OperatorError, OperatorResult, QuantType,
};
use super::trait_impl::{Backend, TensorRef};

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
            CoreOperator::FlashAttention { .. } => true,
            CoreOperator::MoERouter { .. } | CoreOperator::MoEExpert { .. } => {
                true
            }
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

                let batch = input.len() / in_features;
                let m = batch;
                let n = *out_features;
                let k = *in_features;

                // For now, use scalar matmul for F32
                // Real implementation would use gemm_dispatch with weights tensor
                match quant_type {
                    QuantType::F32 => {
                        let weights = inputs[1].as_f32().ok_or_else(|| {
                            OperatorError::InternalError(
                                "Weights is not F32".into(),
                            )
                        })?;
                        // Simple matmul: C = A @ B^T
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0f32;
                                for l in 0..k {
                                    sum +=
                                        input[i * k + l] * weights[j * k + l];
                                }
                                output[i * n + j] = sum;
                            }
                        }
                    }
                    _ => {
                        return Err(OperatorError::UnsupportedOperation(
                            format!(
                                "Quantized Linear not yet supported in ArmCpuBackend: {:?}",
                                quant_type
                            ),
                        ));
                    }
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

                // SwiGLU: output = gate * silu(up)
                for i in 0..*intermediate_size {
                    let silu = up[i] / (1.0 + (-up[i]).exp());
                    output[i] = gate[i] * silu;
                }
                Ok(())
            }

            CoreOperator::FlashAttention { .. } => {
                // Flash Attention requires KV cache context
                // This is a simplified placeholder
                Err(OperatorError::UnsupportedOperation(
                    "FlashAttention requires KV cache context".into(),
                ))
            }

            CoreOperator::Softmax { size, .. } => {
                let input = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Input is not F32".into())
                })?;
                let output = outputs[0].as_f32_mut().ok_or_else(|| {
                    OperatorError::InternalError("Output is not F32".into())
                })?;

                // Softmax
                let n_rows = input.len() / size;
                for row in 0..n_rows {
                    let offset = row * size;
                    let row_input = &input[offset..offset + size];
                    let row_output = &mut output[offset..offset + size];

                    // Find max for numerical stability
                    let max = row_input
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);

                    // Compute exp and sum
                    let mut sum = 0.0f32;
                    for (i, &x) in row_input.iter().enumerate() {
                        row_output[i] = (x - max).exp();
                        sum += row_output[i];
                    }

                    // Normalize
                    let inv_sum = 1.0 / sum;
                    for x in row_output.iter_mut() {
                        *x *= inv_sum;
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

                for i in 0..a.len() {
                    output[i] = a[i] + b[i];
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

                for i in 0..a.len() {
                    output[i] = a[i] * b[i];
                }
                Ok(())
            }

            CoreOperator::MoERouter { n_experts, top_k, .. } => {
                let hidden = inputs[0].as_f32().ok_or_else(|| {
                    OperatorError::InternalError("Hidden is not F32".into())
                })?;

                // Simplified MoE router - just returns top-k indices
                // Real implementation would compute routing weights
                let _ = (hidden, n_experts, top_k);
                Err(OperatorError::UnsupportedOperation(
                    "MoERouter not yet implemented".into(),
                ))
            }

            CoreOperator::MoEExpert { .. } => {
                Err(OperatorError::UnsupportedOperation(
                    "MoEExpert not yet implemented".into(),
                ))
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
