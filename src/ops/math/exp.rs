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

pub fn exp<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> =
        a.data().par_iter().map(|&x| T::from_f32(x.to_f32().exp())).collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn exp_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // ∂L/∂a = ∂L/∂output * exp(a)
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        grad.data_mut()[i] =
            T::from_f32(grad.data()[i].to_f32() * a.data()[i].to_f32().exp());
    }
    vec![grad]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_exp(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let x_fp = (x as f32 - zero) * scale;
            x_fp.exp()
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

pub fn quantized_exp_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        grad.data_mut()[i] = grad.data()[i].saturating_mul(a.data()[i]);
    }
    vec![grad]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct ExpOp;

impl<T: DType + Send + Sync> Operator<T> for ExpOp {
    fn name(&self) -> &'static str {
        "exp"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        exp(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        exp_backward(grad, inputs[0])
    }
}

pub struct QuantizedExpOp;

impl Operator<i8> for QuantizedExpOp {
    fn name(&self) -> &'static str {
        "quantized_exp"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_exp(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_exp_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_f32() {
        let a = Tensor::new(vec![0.0, 1.0, 2.0], &[3]);
        let c = exp(&a);
        let eps = 1e-6;
        assert!(f32::abs(c.data()[0] - 1.0) < eps);
        assert!(f32::abs(c.data()[1] - 2.718281) < eps);
        assert!(f32::abs(c.data()[2] - 7.389056) < eps);
    }
}
