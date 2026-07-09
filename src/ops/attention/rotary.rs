// InferTrain - A Unified Inference and Training Engine
//
// Copyright (c) 2026 Jia Liu & InferTrain Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of InferTrain.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Float Generic Forward (RoPE)
// ============================================================

pub fn rotary_embedding<T: DType + Send + Sync>(
    x: &Tensor<T>,
    cos: &Tensor<T>,
    sin: &Tensor<T>,
) -> Tensor<T> {
    let shape = x.shape();
    assert!(shape.len() >= 2, "rotary_embedding: input must be at least 2D");

    let last_dim = shape[shape.len() - 1];
    let half_dim = last_dim / 2;
    let outer: usize = shape[..shape.len() - 1].iter().product();

    let x_data = x.data();
    let cos_data = cos.data();
    let sin_data = sin.data();

    let mut out_data = vec![T::from_f32(0.0); x.len()];

    for o in 0..outer {
        let base = o * last_dim;
        let cos_val = cos_data[o % cos.len()].to_f32();
        let sin_val = sin_data[o % sin.len()].to_f32();

        for i in 0..half_dim {
            let idx1 = base + i;
            let idx2 = base + i + half_dim;
            let x1 = x_data[idx1].to_f32();
            let x2 = x_data[idx2].to_f32();

            // Rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
            out_data[idx1] = T::from_f32(x1 * cos_val - x2 * sin_val);
            out_data[idx2] = T::from_f32(x1 * sin_val + x2 * cos_val);
        }
    }

    Tensor::new(out_data, shape)
}

// ============================================================
// Float Generic Backward - Simplified TODO: Improve
// ============================================================

pub fn rotary_embedding_backward<T: DType>(
    grad_output: &Tensor<T>,
    _x: &Tensor<T>,
    _cos: &Tensor<T>,
    _sin: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct RotaryOp;

impl<T: DType + Send + Sync> Operator<T> for RotaryOp {
    fn name(&self) -> &'static str {
        "rotary_embedding"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 3);
        rotary_embedding(inputs[0], inputs[1], inputs[2])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        rotary_embedding_backward(grad, inputs[0], inputs[1], inputs[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_embedding() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let cos = Tensor::new(vec![0.5, 0.5], &[2]);
        let sin = Tensor::new(vec![0.8, 0.8], &[2]);
        let c = rotary_embedding(&x, &cos, &sin);
        assert_eq!(c.shape(), &[1, 4]);
    }
}
