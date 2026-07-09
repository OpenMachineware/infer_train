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

pub fn div<T: DType + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in div");

    let data: Vec<T> = a
        .data()
        .par_iter()
        .zip(b.data().par_iter())
        .map(|(&x, &y)| {
            let y_f32 = y.to_f32();
            if y_f32 == 0.0 {
                T::from_f32(f32::INFINITY)
            } else {
                T::from_f32(x.to_f32() / y_f32)
            }
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn div_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for i in 0..grad_a.len() {
        let b_f32 = b.data()[i].to_f32();
        if b_f32 == 0.0 {
            grad_a.data_mut()[i] = T::from_f32(0.0);
            grad_b.data_mut()[i] = T::from_f32(0.0);
        } else {
            grad_a.data_mut()[i] =
                T::from_f32(grad_a.data()[i].to_f32() / b_f32);
            grad_b.data_mut()[i] = T::from_f32(
                -grad_b.data()[i].to_f32() * a.data()[i].to_f32()
                    / (b_f32 * b_f32),
            );
        }
    }
    vec![grad_a, grad_b]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_div(a: &Tensor<i8>, b: &Tensor<i8>) -> Tensor<i8> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in quantized_div");

    let scale_a = a.scale().unwrap_or(1.0);
    let zero_a = a.zero_point().unwrap_or(0.0);
    let scale_b = b.scale().unwrap_or(1.0);
    let zero_b = b.zero_point().unwrap_or(0.0);

    let scale = scale_a;
    let zero = zero_a;

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| {
            let x_fp = (x as f32 - zero_a) * scale_a;
            let y_fp = (y as f32 - zero_b) * scale_b;
            if y_fp == 0.0 {
                f32::INFINITY
            } else {
                x_fp / y_fp
            }
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| {
            if v.is_infinite() {
                127
            } else {
                ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8
            }
        })
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// Quantized Backward
// ============================================================

pub fn quantized_div_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let mut grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for i in 0..grad_a.len() {
        if b.data()[i] == 0 {
            grad_a.data_mut()[i] = 0;
            grad_b.data_mut()[i] = 0;
        } else {
            grad_a.data_mut()[i] = grad_a.data()[i].saturating_div(b.data()[i]);
            grad_b.data_mut()[i] = -grad_b.data()[i]
                .saturating_mul(a.data()[i])
                / (b.data()[i] * b.data()[i]);
        }
    }
    vec![grad_a, grad_b]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct DivOp;

impl<T: DType + Send + Sync> Operator<T> for DivOp {
    fn name(&self) -> &'static str {
        "div"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        div(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        div_backward(grad, inputs[0], inputs[1])
    }
}

pub struct QuantizedDivOp;

impl Operator<i8> for QuantizedDivOp {
    fn name(&self) -> &'static str {
        "quantized_div"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 2);
        quantized_div(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 2);
        quantized_div_backward(grad, inputs[0], inputs[1])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_f32() {
        let a = Tensor::new(vec![4.0, 10.0, 18.0], &[3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], &[3]);
        let c = div(&a, &b);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }
}
