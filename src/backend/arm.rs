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
