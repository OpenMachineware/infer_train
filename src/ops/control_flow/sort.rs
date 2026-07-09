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
// Sort Forward (Sort along specified dimension)
// ============================================================

pub fn sort<T: DType + Send + Sync>(
    input: &Tensor<T>,
    dim: usize,
    ascending: bool,
) -> (Tensor<T>, Tensor<i64>) {
    let shape = input.shape();
    assert!(dim < shape.len(), "sort: dim out of range");

    let dim_size = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let stride = dim_size * inner;

    let input_data = input.data();

    let mut out_values = vec![T::from_f32(0.0); input.len()];
    let mut out_indices = vec![0i64; input.len()];

    for o in 0..outer {
        for i in 0..inner {
            // Collect all values in this dimension
            let mut pairs: Vec<(f32, usize)> = Vec::with_capacity(dim_size);
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                pairs.push((input_data[idx].to_f32(), d));
            }

            // Sort
            if ascending {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            } else {
                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }

            // Write results
            for d in 0..dim_size {
                let idx = base + d * inner;
                out_values[idx] = T::from_f32(pairs[d].0);
                out_indices[idx] = pairs[d].1 as i64;
            }
        }
    }

    (Tensor::new(out_values, shape), Tensor::new(out_indices, shape))
}

// ============================================================
// Sort Backward - Simplified  TODO: Improve
// ============================================================

pub fn sort_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct SortOp;

impl<T: DType + Send + Sync> Operator<T> for SortOp {
    fn name(&self) -> &'static str {
        "sort"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let ascending = attrs.get_bool("ascending").unwrap_or(true);
        let (values, _indices) = sort(inputs[0], dim, ascending);
        values
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        sort_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_1d_ascending() {
        let input = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6]);
        let (values, indices) = sort(&input, 0, true);
        assert_eq!(values.data(), &[1.0, 1.0, 3.0, 4.0, 5.0, 9.0]);
        assert_eq!(indices.data(), &[1, 3, 0, 2, 4, 5]);
    }

    #[test]
    fn test_sort_1d_descending() {
        let input = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6]);
        let (values, _indices) = sort(&input, 0, false);
        assert_eq!(values.data(), &[9.0, 5.0, 4.0, 3.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sort_2d() {
        let input = Tensor::new(
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0],
            &[3, 3],
        );
        let (values, _indices) = sort(&input, 1, true);
        assert_eq!(values.shape(), &[3, 3]);
        assert_eq!(values.data()[0], 1.0);
        assert_eq!(values.data()[1], 3.0);
        assert_eq!(values.data()[2], 4.0);
    }
}
