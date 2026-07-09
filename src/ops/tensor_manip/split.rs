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
// Generic Forward
// ============================================================

pub fn split<T: DType + Send + Sync>(
    input: &Tensor<T>,
    split_size: usize,
    dim: usize,
) -> Vec<Tensor<T>> {
    let shape = input.shape();
    let dim_size = shape[dim];
    assert!(
        dim_size % split_size == 0,
        "split: dim size must be divisible by split size"
    );

    let num_splits = dim_size / split_size;
    let mut result = Vec::with_capacity(num_splits);

    let mut out_shape = shape.to_vec();
    out_shape[dim] = split_size;

    let inner_stride = shape[dim + 1..].iter().product::<usize>();
    let outer = shape[..dim].iter().product::<usize>();
    let dim_stride = inner_stride * shape[dim];

    for s in 0..num_splits {
        let mut data = Vec::with_capacity(outer * split_size * inner_stride);
        for o in 0..outer {
            let base = o * dim_stride + s * split_size * inner_stride;
            for i in 0..split_size {
                let idx = base + i * inner_stride;
                data.extend_from_slice(&input.data()[idx..idx + inner_stride]);
            }
        }
        result.push(Tensor::new(data, &out_shape));
    }

    result
}

// ============================================================
// Generic Backward
// ============================================================

pub fn split_backward<T: DType>(
    grad_outputs: &[&Tensor<T>],
    original_shape: &[usize],
    split_size: usize,
    dim: usize,
) -> Vec<Tensor<T>> {
    let mut data = vec![T::from_f32(0.0); original_shape.iter().product()];
    let inner_stride = original_shape[dim + 1..].iter().product::<usize>();
    let outer = original_shape[..dim].iter().product::<usize>();
    let dim_size = original_shape[dim];
    let dim_stride = inner_stride * dim_size;

    for (s, grad) in grad_outputs.iter().enumerate() {
        for o in 0..outer {
            for i in 0..split_size {
                let src_idx = (o * split_size + i) * inner_stride;
                let dst_idx =
                    o * dim_stride + (s * split_size + i) * inner_stride;
                for j in 0..inner_stride {
                    data[dst_idx + j] = grad.data()[src_idx + j];
                }
            }
        }
    }

    vec![Tensor::new(data, original_shape)]
}

// ============================================================
// Quantized Version (similar pattern, detailed implementation omitted)
// ============================================================

pub fn quantized_split(
    input: &Tensor<i8>,
    split_size: usize,
    dim: usize,
) -> Vec<Tensor<i8>> {
    let shape = input.shape();
    let dim_size = shape[dim];
    assert!(
        dim_size % split_size == 0,
        "quantized_split: dim size must be divisible by split size"
    );

    let num_splits = dim_size / split_size;
    let mut result = Vec::with_capacity(num_splits);

    let mut out_shape = shape.to_vec();
    out_shape[dim] = split_size;

    let inner_stride = shape[dim + 1..].iter().product::<usize>();
    let outer = shape[..dim].iter().product::<usize>();
    let dim_stride = inner_stride * shape[dim];
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);

    for s in 0..num_splits {
        let mut data = Vec::with_capacity(outer * split_size * inner_stride);
        for o in 0..outer {
            let base = o * dim_stride + s * split_size * inner_stride;
            for i in 0..split_size {
                let idx = base + i * inner_stride;
                data.extend_from_slice(&input.data()[idx..idx + inner_stride]);
            }
        }
        result.push(Tensor::<i8>::new_quantized(
            data, &out_shape, scale, zero_point,
        ));
    }

    result
}

pub fn quantized_split_backward(
    grad_outputs: &[&Tensor<i8>],
    original_shape: &[usize],
    split_size: usize,
    dim: usize,
) -> Vec<Tensor<i8>> {
    let mut data = vec![0i8; original_shape.iter().product()];
    let inner_stride = original_shape[dim + 1..].iter().product::<usize>();
    let outer = original_shape[..dim].iter().product::<usize>();
    let dim_size = original_shape[dim];
    let dim_stride = inner_stride * dim_size;
    let scale = grad_outputs[0].scale().unwrap_or(1.0);
    let zero_point = grad_outputs[0].zero_point().unwrap_or(0.0);

    for (s, grad) in grad_outputs.iter().enumerate() {
        for o in 0..outer {
            for i in 0..split_size {
                let src_idx = (o * split_size + i) * inner_stride;
                let dst_idx =
                    o * dim_stride + (s * split_size + i) * inner_stride;
                for j in 0..inner_stride {
                    data[dst_idx + j] = grad.data()[src_idx + j];
                }
            }
        }
    }

    vec![Tensor::<i8>::new_quantized(data, original_shape, scale, zero_point)]
}

// ============================================================
// Operator Trait
// ============================================================

pub struct SplitOp;

impl<T: DType + Send + Sync> Operator<T> for SplitOp {
    fn name(&self) -> &'static str {
        "split"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let split_size =
            attrs.get_int("split_size").map(|v| v as usize).unwrap_or(1);
        let dim = attrs.get_int("dim").map(|v| v as usize).unwrap_or(0);
        let results = split(inputs[0], split_size, dim);
        results.into_iter().next().unwrap_or_else(|| inputs[0].clone())
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        let split_size =
            attrs.get_int("split_size").map(|v| v as usize).unwrap_or(1);
        let dim = attrs.get_int("dim").map(|v| v as usize).unwrap_or(0);
        split_backward(&[grad], inputs[0].shape(), split_size, dim)
    }
}

pub struct QuantizedSplitOp;

impl Operator<i8> for QuantizedSplitOp {
    fn name(&self) -> &'static str {
        "quantized_split"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let split_size =
            attrs.get_int("split_size").map(|v| v as usize).unwrap_or(1);
        let dim = attrs.get_int("dim").map(|v| v as usize).unwrap_or(0);
        let results = quantized_split(inputs[0], split_size, dim);
        results.into_iter().next().unwrap_or_else(|| inputs[0].clone())
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        let split_size =
            attrs.get_int("split_size").map(|v| v as usize).unwrap_or(1);
        let dim = attrs.get_int("dim").map(|v| v as usize).unwrap_or(0);
        quantized_split_backward(&[grad], inputs[0].shape(), split_size, dim)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}
