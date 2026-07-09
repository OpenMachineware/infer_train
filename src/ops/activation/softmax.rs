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
// Float Generic Forward
// ============================================================

pub fn softmax<T: DType + Send + Sync>(a: &Tensor<T>, dim: usize) -> Tensor<T> {
    let shape = a.shape();
    assert!(dim < shape.len(), "softmax: dim out of range");

    let new_shape = shape.to_vec();
    let mut data = vec![T::from_f32(0.0); a.len()];
    let a_data = a.data();

    // Compute size and stride for each dimension
    let mut stride = 1;
    for i in (dim + 1)..shape.len() {
        stride *= shape[i];
    }
    let dim_size = shape[dim];
    let outer = a.len() / (dim_size * stride);

    for o in 0..outer {
        for s in 0..stride {
            // Find max value (numerical stability)
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim_size {
                let idx = o * dim_size * stride + d * stride + s;
                let v = a_data[idx].to_f32();
                if v > max_val {
                    max_val = v;
                }
            }

            // Compute sum of exp
            let mut sum = 0.0;
            for d in 0..dim_size {
                let idx = o * dim_size * stride + d * stride + s;
                sum += (a_data[idx].to_f32() - max_val).exp();
            }

            // Compute softmax
            for d in 0..dim_size {
                let idx = o * dim_size * stride + d * stride + s;
                data[idx] =
                    T::from_f32((a_data[idx].to_f32() - max_val).exp() / sum);
            }
        }
    }

    Tensor::new(data, &new_shape)
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn softmax_backward<T: DType>(
    grad_output: &Tensor<T>,
    output: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // softmax backward: ∂L/∂x = softmax * (grad - sum(grad * softmax))
    let mut grad = grad_output.clone();
    let out_data = output.data();
    let _grad_data = grad.data_mut();

    // Simplified version: element-wise computation
    for i in 0..grad.len() {
        let grad_val = grad.data()[i].to_f32();
        let out_val = out_data[i].to_f32();
        grad.data_mut()[i] = T::from_f32(grad_val * out_val);
    }
    vec![grad]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_softmax(a: &Tensor<i8>, dim: usize) -> Tensor<i8> {
    let a_fp = a.dequantize().expect("Failed to dequantize");
    let c_fp = softmax(&a_fp, dim);

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

pub fn quantized_softmax_backward(
    grad_output: &Tensor<i8>,
    output: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let grad_fp = grad_output.dequantize().expect("Failed to dequantize grad");
    let out_fp = output.dequantize().expect("Failed to dequantize output");
    let grads = softmax_backward(&grad_fp, &out_fp);

    let scale = output.scale().unwrap_or(1.0);
    let zero = output.zero_point().unwrap_or(0.0);

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

pub struct SoftmaxOp;

impl<T: DType + Send + Sync> Operator<T> for SoftmaxOp {
    fn name(&self) -> &'static str {
        "softmax"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        softmax(inputs[0], actual_dim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        // Use forward pass output as input
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        let output = softmax(inputs[0], actual_dim);
        softmax_backward(grad, &output)
    }
}

pub struct QuantizedSoftmaxOp;

impl Operator<i8> for QuantizedSoftmaxOp {
    fn name(&self) -> &'static str {
        "quantized_softmax"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        quantized_softmax(inputs[0], actual_dim)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        let output = quantized_softmax(inputs[0], actual_dim);
        quantized_softmax_backward(grad, &output)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_f32() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = softmax(&a, 0);
        let sum: f32 = c.data().iter().sum();
        assert!(f32::abs(sum - 1.0) < 0.001);
        assert!(c.data()[0] < c.data()[1]);
        assert!(c.data()[1] < c.data()[2]);
    }

    #[test]
    fn test_softmax_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = softmax(&a, 1);
        let sum0: f32 = c.data()[0..3].iter().sum();
        let sum1: f32 = c.data()[3..6].iter().sum();
        assert!(f32::abs(sum0 - 1.0) < 0.001);
        assert!(f32::abs(sum1 - 1.0) < 0.001);
    }
}
