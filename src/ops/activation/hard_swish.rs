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

pub fn hard_swish<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            // hard_swish = x * ReLU6(x + 3) / 6
            let relu6 = if v + 3.0 > 0.0 {
                if v + 3.0 < 6.0 {
                    v + 3.0
                } else {
                    6.0
                }
            } else {
                0.0
            };
            T::from_f32(v * relu6 / 6.0)
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn hard_swish_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        // Derivative: (2*x + 3)/6 when -3 < x < 3, otherwise 0 or 1
        let dhs = if v < -3.0 {
            0.0
        } else if v > 3.0 {
            1.0
        } else {
            (2.0 * v + 3.0) / 6.0
        };
        grad.data_mut()[i] = T::from_f32(grad.data()[i].to_f32() * dhs);
    }
    vec![grad]
}

// ============================================================
// Quantized Forward - Simplified   TODO: Improve
// ============================================================

pub fn quantized_hard_swish(a: &Tensor<i8>) -> Tensor<i8> {
    let a_fp = a.dequantize().expect("Failed to dequantize");
    let c_fp = hard_swish(&a_fp);

    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = c_fp
        .data()
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, c_fp.shape(), scale, zero)
}

// ============================================================
// Quantized Backward - Simplified   TODO: Improve
// ============================================================

pub fn quantized_hard_swish_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let grad_fp = grad_output.dequantize().expect("Failed to dequantize grad");
    let a_fp = a.dequantize().expect("Failed to dequantize a");
    let grads = hard_swish_backward(&grad_fp, &a_fp);

    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = grads[0]
        .data()
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    vec![Tensor::<i8>::new_quantized(data, grads[0].shape(), scale, zero)]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct HardSwishOp;

impl<T: DType + Send + Sync> Operator<T> for HardSwishOp {
    fn name(&self) -> &'static str {
        "hard_swish"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        hard_swish(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        hard_swish_backward(grad, inputs[0])
    }
}

pub struct QuantizedHardSwishOp;

impl Operator<i8> for QuantizedHardSwishOp {
    fn name(&self) -> &'static str {
        "quantized_hard_swish"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_hard_swish(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_hard_swish_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_swish_f32() {
        let a = Tensor::new(vec![-4.0, -2.0, 0.0, 2.0, 4.0], &[5]);
        let c = hard_swish(&a);
        let eps = 0.01;
        assert!(f32::abs(c.data()[0] - 0.0) < eps);
        assert!(f32::abs(c.data()[1] + 0.333) < eps);
        assert!(f32::abs(c.data()[2] - 0.0) < eps);
        assert!(f32::abs(c.data()[3] - 1.667) < eps);
        assert!(f32::abs(c.data()[4] - 4.0) < eps);
    }
}
