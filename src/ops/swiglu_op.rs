// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// SwiGLU activation operator

use crate::backend::operator::{Operator, OperatorId};
use crate::memory::RequestScratch;
use std::fmt;

/// SwiGLU activation operator
///
/// Computes: output = gate * silu(up)
/// where silu(x) = x / (1 + exp(-x))
pub struct SwiGLUOp {
    id: OperatorId,
    intermediate_size: usize,
}

impl SwiGLUOp {
    /// Create a new SwiGLU operator
    pub fn new(id: OperatorId, intermediate_size: usize) -> Self {
        Self { id, intermediate_size }
    }

    /// Execute SwiGLU activation
    ///
    /// gate and up are modified in-place, output is written to gate
    #[cfg(target_arch = "aarch64")]
    pub fn execute(
        &self,
        gate: &mut [f32],
        up: &[f32],
        _scratch: &mut RequestScratch,
    ) {
        debug_assert_eq!(gate.len(), self.intermediate_size);
        debug_assert_eq!(up.len(), self.intermediate_size);

        // SwiGLU: gate = gate * silu(up)
        for i in 0..self.intermediate_size {
            let silu = up[i] / (1.0 + (-up[i]).exp());
            gate[i] = gate[i] * silu;
        }
    }
}

impl Operator for SwiGLUOp {
    fn id(&self) -> OperatorId {
        self.id
    }

    fn name(&self) -> &str {
        "SwiGLU"
    }

    fn input_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.intermediate_size], vec![self.intermediate_size]]
    }

    fn output_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.intermediate_size]]
    }
}

impl fmt::Debug for SwiGLUOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SwiGLUOp")
            .field("id", &self.id)
            .field("intermediate_size", &self.intermediate_size)
            .finish()
    }
}
