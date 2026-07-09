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

pub fn reshape<T: DType + Send + Sync>(
    input: &Tensor<T>,
    new_shape: &[usize],
) -> Tensor<T> {
    let total: usize = input.shape().iter().product();
    let new_total: usize = new_shape.iter().product();
    assert_eq!(total, new_total, "reshape: total elements must match");
    Tensor::new(input.data().to_vec(), new_shape)
}

// ============================================================
// Generic Backward
// ============================================================

pub fn reshape_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
) -> Vec<Tensor<T>> {
    vec![reshape(grad_output, original_shape)]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_reshape(
    input: &Tensor<i8>,
    new_shape: &[usize],
) -> Tensor<i8> {
    let total: usize = input.shape().iter().product();
    let new_total: usize = new_shape.iter().product();
    assert_eq!(
        total, new_total,
        "quantized_reshape: total elements must match"
    );
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(
        input.data().to_vec(),
        new_shape,
        scale,
        zero_point,
    )
}

// ============================================================
// Quantized Backward
// ============================================================

pub fn quantized_reshape_backward(
    grad_output: &Tensor<i8>,
    original_shape: &[usize],
) -> Vec<Tensor<i8>> {
    vec![quantized_reshape(grad_output, original_shape)]
}

// ============================================================
// Operator Trait
// ============================================================

pub struct ReshapeOp;

impl<T: DType + Send + Sync> Operator<T> for ReshapeOp {
    fn name(&self) -> &'static str {
        "reshape"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let shape = attrs
            .get_int_list("shape")
            .expect("reshape requires shape")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        reshape(inputs[0], &shape)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        reshape_backward(grad, inputs[0].shape())
    }
}

pub struct QuantizedReshapeOp;

impl Operator<i8> for QuantizedReshapeOp {
    fn name(&self) -> &'static str {
        "quantized_reshape"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let shape = attrs
            .get_int_list("shape")
            .expect("quantized_reshape requires shape")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        quantized_reshape(inputs[0], &shape)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        quantized_reshape_backward(grad, inputs[0].shape())
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}
