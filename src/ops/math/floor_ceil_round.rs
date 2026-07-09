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
// FLOOR
// ============================================================

pub fn floor<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> =
        a.data().par_iter().map(|&x| T::from_f32(x.to_f32().floor())).collect();
    Tensor::new(data, a.shape())
}

pub fn floor_backward<T: DType>(
    grad_output: &Tensor<T>,
    _a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![Tensor::zeros(grad_output.shape())]
}

pub fn quantized_floor(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            v.floor()
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

pub struct FloorOp;

impl<T: DType + Send + Sync> Operator<T> for FloorOp {
    fn name(&self) -> &'static str {
        "floor"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        floor(inputs[0])
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        floor_backward(_grad, inputs[0])
    }
}

pub struct QuantizedFloorOp;

impl Operator<i8> for QuantizedFloorOp {
    fn name(&self) -> &'static str {
        "quantized_floor"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_floor(inputs[0])
    }
    fn backward(
        &self,
        _grad: &Tensor<i8>,
        _inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        vec![Tensor::zeros(_inputs[0].shape())]
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

// ============================================================
// CEIL
// ============================================================

pub fn ceil<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> =
        a.data().par_iter().map(|&x| T::from_f32(x.to_f32().ceil())).collect();
    Tensor::new(data, a.shape())
}

pub fn ceil_backward<T: DType>(
    grad_output: &Tensor<T>,
    _a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![Tensor::zeros(grad_output.shape())]
}

pub fn quantized_ceil(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            v.ceil()
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

pub struct CeilOp;

impl<T: DType + Send + Sync> Operator<T> for CeilOp {
    fn name(&self) -> &'static str {
        "ceil"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        ceil(inputs[0])
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        ceil_backward(_grad, inputs[0])
    }
}

pub struct QuantizedCeilOp;

impl Operator<i8> for QuantizedCeilOp {
    fn name(&self) -> &'static str {
        "quantized_ceil"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_ceil(inputs[0])
    }
    fn backward(
        &self,
        _grad: &Tensor<i8>,
        _inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        vec![Tensor::zeros(_inputs[0].shape())]
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

// ============================================================
// ROUND
// ============================================================

pub fn round<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> =
        a.data().par_iter().map(|&x| T::from_f32(x.to_f32().round())).collect();
    Tensor::new(data, a.shape())
}

pub fn round_backward<T: DType>(
    grad_output: &Tensor<T>,
    _a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![Tensor::zeros(grad_output.shape())]
}

pub fn quantized_round(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            v.round()
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

pub struct RoundOp;

impl<T: DType + Send + Sync> Operator<T> for RoundOp {
    fn name(&self) -> &'static str {
        "round"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        round(inputs[0])
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        round_backward(_grad, inputs[0])
    }
}

pub struct QuantizedRoundOp;

impl Operator<i8> for QuantizedRoundOp {
    fn name(&self) -> &'static str {
        "quantized_round"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_round(inputs[0])
    }
    fn backward(
        &self,
        _grad: &Tensor<i8>,
        _inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        vec![Tensor::zeros(_inputs[0].shape())]
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_f32() {
        let a = Tensor::new(vec![1.9, -1.1, 2.0], &[3]);
        let c = floor(&a);
        assert_eq!(c.data(), &[1.0, -2.0, 2.0]);
    }

    #[test]
    fn test_ceil_f32() {
        let a = Tensor::new(vec![1.1, -1.9, 2.0], &[3]);
        let c = ceil(&a);
        assert_eq!(c.data(), &[2.0, -1.0, 2.0]);
    }

    #[test]
    fn test_round_f32() {
        let a = Tensor::new(vec![1.4, 1.6, -1.4, -1.6], &[4]);
        let c = round(&a);
        assert_eq!(c.data(), &[1.0, 2.0, -1.0, -2.0]);
    }
}
