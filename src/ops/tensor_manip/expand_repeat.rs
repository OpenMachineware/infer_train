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
// Expand Generic Forward
// ============================================================

pub fn expand<T: DType + Send + Sync>(
    input: &Tensor<T>,
    target_shape: &[usize],
) -> Tensor<T> {
    let shape = input.shape();
    assert!(
        target_shape.len() >= shape.len(),
        "expand: target shape must have same or more dims"
    );

    let mut data = Vec::new();
    let total = target_shape.iter().product::<usize>();
    for _ in 0..total {
        data.extend_from_slice(input.data());
    }
    Tensor::new(data, target_shape)
}

// ============================================================
// Expand Generic Backward
// ============================================================

pub fn expand_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
) -> Vec<Tensor<T>> {
    vec![Tensor::new(grad_output.data().to_vec(), original_shape)]
}

// ============================================================
// Repeat Generic Forward
// ============================================================

pub fn repeat<T: DType + Send + Sync>(
    input: &Tensor<T>,
    repeats: &[usize],
) -> Tensor<T> {
    let shape = input.shape();
    assert_eq!(shape.len(), repeats.len(), "repeat: repeats must match rank");

    let mut data = Vec::new();
    let total = repeats.iter().product::<usize>();
    for _ in 0..total {
        data.extend_from_slice(input.data());
    }
    let new_shape: Vec<usize> =
        shape.iter().zip(repeats).map(|(a, b)| a * b).collect();
    Tensor::new(data, &new_shape)
}

// ============================================================
// Repeat Generic Backward
// ============================================================

pub fn repeat_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
) -> Vec<Tensor<T>> {
    let len: usize = original_shape.iter().product();
    vec![Tensor::new(grad_output.data()[..len].to_vec(), original_shape)]
}

// ============================================================
// Quantized Expand Forward
// ============================================================

pub fn quantized_expand(
    input: &Tensor<i8>,
    target_shape: &[usize],
) -> Tensor<i8> {
    let mut data = Vec::new();
    let total = target_shape.iter().product::<usize>();
    for _ in 0..total {
        data.extend_from_slice(input.data());
    }
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(data, target_shape, scale, zero_point)
}

// ============================================================
// Quantized Expand Backward
// ============================================================

pub fn quantized_expand_backward(
    grad_output: &Tensor<i8>,
    original_shape: &[usize],
) -> Vec<Tensor<i8>> {
    let scale = grad_output.scale().unwrap_or(1.0);
    let zero_point = grad_output.zero_point().unwrap_or(0.0);
    vec![Tensor::<i8>::new_quantized(
        grad_output.data().to_vec(),
        original_shape,
        scale,
        zero_point,
    )]
}

// ============================================================
// Quantized Repeat Forward
// ============================================================

pub fn quantized_repeat(input: &Tensor<i8>, repeats: &[usize]) -> Tensor<i8> {
    let shape = input.shape();
    assert_eq!(
        shape.len(),
        repeats.len(),
        "quantized_repeat: repeats must match rank"
    );

    let mut data = Vec::new();
    let total = repeats.iter().product::<usize>();
    for _ in 0..total {
        data.extend_from_slice(input.data());
    }
    let new_shape: Vec<usize> =
        shape.iter().zip(repeats).map(|(a, b)| a * b).collect();
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(data, &new_shape, scale, zero_point)
}

// ============================================================
// Quantized Repeat Backward
// ============================================================

pub fn quantized_repeat_backward(
    grad_output: &Tensor<i8>,
    original_shape: &[usize],
) -> Vec<Tensor<i8>> {
    let len: usize = original_shape.iter().product();
    let scale = grad_output.scale().unwrap_or(1.0);
    let zero_point = grad_output.zero_point().unwrap_or(0.0);
    vec![Tensor::<i8>::new_quantized(
        grad_output.data()[..len].to_vec(),
        original_shape,
        scale,
        zero_point,
    )]
}

// ============================================================
// Expand Operator
// ============================================================

pub struct ExpandOp;

impl<T: DType + Send + Sync> Operator<T> for ExpandOp {
    fn name(&self) -> &'static str {
        "expand"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let target_shape = attrs
            .get_int_list("shape")
            .expect("expand requires shape")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        expand(inputs[0], &target_shape)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        expand_backward(grad, inputs[0].shape())
    }
}

// ============================================================
// Repeat Operator
// ============================================================

pub struct RepeatOp;

impl<T: DType + Send + Sync> Operator<T> for RepeatOp {
    fn name(&self) -> &'static str {
        "repeat"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let repeats = attrs
            .get_int_list("repeats")
            .expect("repeat requires repeats")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        repeat(inputs[0], &repeats)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        repeat_backward(grad, inputs[0].shape())
    }
}

// ============================================================
// Quantized Expand Operator
// ============================================================

pub struct QuantizedExpandOp;

impl Operator<i8> for QuantizedExpandOp {
    fn name(&self) -> &'static str {
        "quantized_expand"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let target_shape = attrs
            .get_int_list("shape")
            .expect("quantized_expand requires shape")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        quantized_expand(inputs[0], &target_shape)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        quantized_expand_backward(grad, inputs[0].shape())
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

// ============================================================
// Quantized Repeat Operator
// ============================================================

pub struct QuantizedRepeatOp;

impl Operator<i8> for QuantizedRepeatOp {
    fn name(&self) -> &'static str {
        "quantized_repeat"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let repeats = attrs
            .get_int_list("repeats")
            .expect("quantized_repeat requires repeats")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        quantized_repeat(inputs[0], &repeats)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        quantized_repeat_backward(grad, inputs[0].shape())
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}
