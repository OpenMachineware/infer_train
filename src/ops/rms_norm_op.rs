// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// RMS Norm operator

use crate::backend::operator::{Operator, OperatorId};
use crate::cpu::CpuWorkBuffers;
use std::fmt;

/// RMS Norm operator
///
/// Computes: output = (input / sqrt(mean(input^2) + eps)) * weight
pub struct RmsNormOp {
    id: OperatorId,
    hidden_size: usize,
    eps: f32,
}

impl RmsNormOp {
    /// Create a new RMS Norm operator
    pub fn new(id: OperatorId, hidden_size: usize, eps: f32) -> Self {
        Self { id, hidden_size, eps }
    }

    /// Execute RMS Norm
    ///
    /// Uses pre-dequantized weights for efficiency
    #[cfg(target_arch = "aarch64")]
    pub fn execute(
        &self,
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        _buffers: &mut CpuWorkBuffers,
    ) {
        debug_assert_eq!(input.len(), self.hidden_size);
        debug_assert_eq!(weight.len(), self.hidden_size);
        debug_assert_eq!(output.len(), self.hidden_size);

        // Compute sum of squares
        let mut sum_sq = 0.0f32;
        for i in 0..self.hidden_size {
            sum_sq += input[i] * input[i];
        }

        // Compute RMS
        let mean_sq = sum_sq / self.hidden_size as f32;
        let rms = (mean_sq + self.eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Normalize and scale
        for i in 0..self.hidden_size {
            output[i] = input[i] * inv_rms * weight[i];
        }
    }
}

impl Operator for RmsNormOp {
    fn id(&self) -> OperatorId {
        self.id
    }

    fn name(&self) -> &str {
        "RmsNorm"
    }

    fn input_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.hidden_size], vec![self.hidden_size]]
    }

    fn output_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.hidden_size]]
    }
}

impl fmt::Debug for RmsNormOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RmsNormOp")
            .field("id", &self.id)
            .field("hidden_size", &self.hidden_size)
            .finish()
    }
}
