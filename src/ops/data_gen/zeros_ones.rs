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
// Zeros
// ============================================================

pub fn zeros<T: DType + Send + Sync>(shape: &[usize]) -> Tensor<T> {
    let size: usize = shape.iter().product();
    let data = vec![T::zero(); size];
    Tensor::new(data, shape)
}

// ============================================================
// Ones
// ============================================================

pub fn ones<T: DType + Send + Sync>(shape: &[usize]) -> Tensor<T> {
    let size: usize = shape.iter().product();
    let data = vec![T::one(); size];
    Tensor::new(data, shape)
}

// ============================================================
// Full (fill with value)
// ============================================================

pub fn full<T: DType + Send + Sync>(shape: &[usize], value: f32) -> Tensor<T> {
    let size: usize = shape.iter().product();
    let data = vec![T::from_f32(value); size];
    Tensor::new(data, shape)
}

// ============================================================
// Operators
// ============================================================

pub struct ZerosOp;

impl<T: DType + Send + Sync> Operator<T> for ZerosOp {
    fn name(&self) -> &'static str {
        "zeros"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // Get shape from input
        let shape = inputs[0].shape();
        zeros::<T>(shape)
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }
}

pub struct OnesOp;

impl<T: DType + Send + Sync> Operator<T> for OnesOp {
    fn name(&self) -> &'static str {
        "ones"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        let shape = inputs[0].shape();
        ones::<T>(shape)
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }
}

pub struct FullOp;

impl<T: DType + Send + Sync> Operator<T> for FullOp {
    fn name(&self) -> &'static str {
        "full"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        let shape = inputs[0].shape();
        let value = attrs.get_float("value").unwrap_or(0.0);
        full::<T>(shape, value)
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let c = zeros::<f32>(&[2, 3]);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let c = ones::<f32>(&[2, 3]);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_full() {
        let c = full::<f32>(&[2, 3], 5.0);
        assert_eq!(c.data(), &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    }
}
