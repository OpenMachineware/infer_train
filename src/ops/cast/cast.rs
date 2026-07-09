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
// Cast (Type Conversion)
// ============================================================

pub fn cast<T: DType + Send + Sync, U: DType + Send + Sync>(
    input: &Tensor<T>,
) -> Tensor<U> {
    let data: Vec<U> =
        input.data().par_iter().map(|&x| U::from_f32(x.to_f32())).collect();

    Tensor::new(data, input.shape())
}

// ============================================================
// Cast Backward (Backpropagation is identity mapping)
// ============================================================

pub fn cast_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct CastOp;

impl<T: DType + Send + Sync> Operator<T> for CastOp {
    fn name(&self) -> &'static str {
        "cast"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        // Return original value (actual cast needs target type)
        inputs[0].clone()
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        cast_backward(grad)
    }
}

// ============================================================
// Convenience Functions: Specific Type Conversions
// ============================================================

pub fn to_f32<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<f32> {
    cast::<T, f32>(input)
}

pub fn to_f64<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<f64> {
    cast::<T, f64>(input)
}

pub fn to_f16<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<half::f16> {
    use half::f16;
    cast::<T, f16>(input)
}

pub fn to_bf16<T: DType + Send + Sync>(
    input: &Tensor<T>,
) -> Tensor<half::bf16> {
    use half::bf16;
    cast::<T, bf16>(input)
}

pub fn to_i8<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<i8> {
    cast::<T, i8>(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_f32_to_f64() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = cast::<f32, f64>(&input);
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_f32() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = to_f32(&input);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cast_f64_to_f32() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = cast::<f64, f32>(&input);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }
}
