// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Metal GPU backend for Apple Silicon.
//!
//! This module is only compiled on aarch64 with metal feature.

use super::device::{Device, DeviceId};
use super::memory::MemoryRegionId;
use super::operator::{
    CoreOperator, Cost, OperatorError, OperatorResult, QuantType,
};
use super::trait_impl::{Backend, TensorRef};

/// Metal GPU backend.
#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
    memory_region: MemoryRegionId,
}

impl MetalBackend {
    /// Create a new Metal backend.
    pub fn new(memory_region: MemoryRegionId) -> Self {
        let device = Device::detect_best();
        Self { device, memory_region }
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "Metal GPU"
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
            | CoreOperator::FlashAttentionDecode { .. }
            | CoreOperator::FlashAttentionPrefill { .. }
            | CoreOperator::SwiGLU { .. }
            | CoreOperator::Softmax { .. } => true,
            CoreOperator::Add { .. } | CoreOperator::Mul { .. } => true,
            CoreOperator::MoERouter { .. }
            | CoreOperator::MoEExpertBatch { .. } => false, // Not yet implemented for GPU
        }
    }

    fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRef],
    ) -> OperatorResult<()> {
        // For now, delegate to CPU implementation
        // Real implementation would use Metal kernels
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

                // CPU fallback
                crate::ops::rms_norm::rms_norm_f32(
                    input,
                    weight,
                    output,
                    *hidden_size,
                    *eps,
                );
                Ok(())
            }

            CoreOperator::FlashAttentionDecode {
                n_heads, head_dim, ..
            } => {
                // Flash Attention decode requires GPU context
                // Placeholder for now
                Err(OperatorError::UnsupportedOperation(
                    "FlashAttentionDecode requires Metal context".into(),
                ))
            }

            CoreOperator::FlashAttentionPrefill {
                n_heads, head_dim, ..
            } => {
                // Flash Attention prefill requires GPU context
                // Placeholder for now
                Err(OperatorError::UnsupportedOperation(
                    "FlashAttentionPrefill requires Metal context".into(),
                ))
            }

            _ => Err(OperatorError::UnsupportedOperation(format!(
                "Operator not yet supported in MetalBackend: {:?}",
                op
            ))),
        }
    }

    fn estimate_cost(&self, op: &CoreOperator) -> Cost {
        match op {
            CoreOperator::RmsNorm { hidden_size, .. } => {
                // GPU is faster for large hidden sizes
                Cost {
                    compute_us: *hidden_size as f64 * 0.0001, // 10x faster than CPU
                    memory_bytes: *hidden_size as u64 * 4 * 3,
                    read_bytes: *hidden_size as u64 * 4 * 2,
                    write_bytes: *hidden_size as u64 * 4,
                }
            }
            CoreOperator::FlashAttentionDecode {
                n_heads, head_dim, ..
            } => {
                // Flash Attention decode is very efficient on GPU
                let flops = 4 * n_heads * head_dim;
                Cost {
                    compute_us: flops as f64 * 0.00001,
                    memory_bytes: (*n_heads * *head_dim) as u64 * 4,
                    read_bytes: (*n_heads * *head_dim * 3) as u64 * 4,
                    write_bytes: (*n_heads * *head_dim) as u64 * 4,
                }
            }
            CoreOperator::FlashAttentionPrefill {
                n_heads, head_dim, ..
            } => {
                // Flash Attention prefill is very efficient on GPU
                let flops = 4 * n_heads * head_dim;
                Cost {
                    compute_us: flops as f64 * 0.00001,
                    memory_bytes: (*n_heads * *head_dim) as u64 * 4,
                    read_bytes: (*n_heads * *head_dim * 3) as u64 * 4,
                    write_bytes: (*n_heads * *head_dim) as u64 * 4,
                }
            }
            _ => Cost::zero(),
        }
    }
}
