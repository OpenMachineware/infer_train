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
use rayon::prelude::*;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn sqrt<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            if v >= 0.0 {
                T::from_f32(v.sqrt())
            } else {
                T::from_f32(0.0)
            }
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn sqrt_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // ∂L/∂a = ∂L/∂output / (2 * sqrt(a))
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        if v > 0.0 {
            grad.data_mut()[i] =
                T::from_f32(grad.data()[i].to_f32() / (2.0 * v.sqrt()));
        } else {
            grad.data_mut()[i] = T::from_f32(0.0);
        }
    }
    vec![grad]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_sqrt(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            if v >= 0.0 {
                v.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// Quantized Backward
// ============================================================

pub fn quantized_sqrt_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        if a.data()[i] > 0 {
            grad.data_mut()[i] = grad.data()[i] / (2 * a.data()[i]);
        } else {
            grad.data_mut()[i] = 0;
        }
    }
    vec![grad]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct SqrtOp;

impl<T: DType + Send + Sync> Operator<T> for SqrtOp {
    fn name(&self) -> &'static str {
        "sqrt"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        sqrt(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        sqrt_backward(grad, inputs[0])
    }
}

pub struct QuantizedSqrtOp;

impl Operator<i8> for QuantizedSqrtOp {
    fn name(&self) -> &'static str {
        "quantized_sqrt"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_sqrt(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_sqrt_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_f32() {
        let a = Tensor::new(vec![4.0, 9.0, 16.0], &[3]);
        let c = sqrt(&a);
        assert_eq!(c.data(), &[2.0, 3.0, 4.0]);
    }
}
