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

pub fn prod<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    let shape = a.shape();
    assert!(dim < shape.len(), "prod: dim out of range");

    let dim_size = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let stride = dim_size * inner;

    let a_data = a.data();
    let out_size = if keepdim {
        let mut new_shape = shape.to_vec();
        new_shape[dim] = 1;
        new_shape.iter().product()
    } else {
        let mut new_shape = shape.to_vec();
        new_shape.remove(dim);
        new_shape.iter().product()
    };

    let mut out_data = vec![T::from_f32(0.0); out_size];

    for o in 0..outer {
        for i in 0..inner {
            let mut prod_val = T::from_f32(1.0);
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                prod_val = prod_val * a_data[idx];
            }
            let out_idx = o * inner + i;
            out_data[out_idx] = prod_val;
        }
    }

    let out_shape = if keepdim {
        let mut new_shape = shape.to_vec();
        new_shape[dim] = 1;
        new_shape
    } else {
        let mut new_shape = shape.to_vec();
        new_shape.remove(dim);
        new_shape
    };

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// Float Generic Backward - Simplified   TODO: Improve
// ============================================================

pub fn prod_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct ProdOp;

impl<T: DType + Send + Sync> Operator<T> for ProdOp {
    fn name(&self) -> &'static str {
        "prod"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        prod(inputs[0], dim, keepdim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        prod_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prod_1d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let c = prod(&a, 0, false);
        assert_eq!(c.data(), &[24.0]);
    }

    #[test]
    fn test_prod_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = prod(&a, 1, false);
        assert_eq!(c.data(), &[6.0, 120.0]);
    }
}
