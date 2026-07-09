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
// use super::matmul::{matmul, matmul_backward};

// ============================================================
// Float Generic Forward
// ============================================================

pub fn batch_matmul<T: DType + Send + Sync>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Tensor<T> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert!(a_shape.len() >= 3, "batch_matmul: a must be at least 3D");
    assert!(b_shape.len() >= 3, "batch_matmul: b must be at least 3D");

    // Batch dimensions must match
    let batch_dims = a_shape.len().max(b_shape.len()) - 2;
    for i in 0..batch_dims {
        let a_dim = if i < a_shape.len() - 2 { a_shape[i] } else { 1 };
        let b_dim = if i < b_shape.len() - 2 { b_shape[i] } else { 1 };
        assert!(
            a_dim == b_dim || a_dim == 1 || b_dim == 1,
            "batch_matmul: batch dimension {} mismatch: {} vs {}",
            i,
            a_dim,
            b_dim
        );
    }

    // Compute output shape
    let mut out_shape = Vec::new();
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();
    for i in 0..batch_dims {
        let a_idx = if i < a_rank - 2 { a_shape[i] } else { 1 };
        let b_idx = if i < b_rank - 2 { b_shape[i] } else { 1 };
        out_shape.push(a_idx.max(b_idx));
    }

    let m = a_shape[a_rank - 2];
    let k_a = a_shape[a_rank - 1];
    let k_b = b_shape[b_rank - 2];
    let n = b_shape[b_rank - 1];
    assert_eq!(
        k_a, k_b,
        "batch_matmul: inner dimensions must match: {} vs {}",
        k_a, k_b
    );

    out_shape.push(m);
    out_shape.push(n);

    // Compute for each batch
    let batch_total: usize = out_shape[..batch_dims].iter().product();
    let out_batch_stride = m * n;
    let _a_batch_stride = a_shape[a_rank - 2] * a_shape[a_rank - 1];
    let _b_batch_stride = b_shape[b_rank - 2] * b_shape[b_rank - 1];

    // Compute start offsets for a and b for each batch
    let a_data = a.data();
    let b_data = b.data();

    let result: Vec<T> = (0..batch_total * out_batch_stride)
        .into_par_iter()
        .map(|flat_idx| {
            let batch_idx = flat_idx / out_batch_stride;
            let local_idx = flat_idx % out_batch_stride;
            let i = local_idx / n;
            let j = local_idx % n;

            // Compute a_offset and b_offset
            let mut a_offset = 0;
            let mut b_offset = 0;
            let mut tmp = batch_idx;
            for dim in (0..batch_dims).rev() {
                let dim_size = out_shape[dim];
                let idx = tmp % dim_size;
                tmp /= dim_size;

                if dim < a_rank - 2 {
                    let a_dim = a_shape[dim];
                    let a_idx = if a_dim == 1 { 0 } else { idx % a_dim };
                    a_offset +=
                        a_idx * a_shape[dim + 1..].iter().product::<usize>();
                }
                if dim < b_rank - 2 {
                    let b_dim = b_shape[dim];
                    let b_idx = if b_dim == 1 { 0 } else { idx % b_dim };
                    b_offset +=
                        b_idx * b_shape[dim + 1..].iter().product::<usize>();
                }
            }

            let mut sum = T::from_f32(0.0);
            for t in 0..k_a {
                let a_idx = a_offset + i * k_a + t;
                let b_idx = b_offset + t * n + j;
                sum = sum + a_data[a_idx] * b_data[b_idx];
            }
            sum
        })
        .collect();

    Tensor::new(result, &out_shape)
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn batch_matmul_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // batch_matmul backward: apply matmul_backward to each batch
    // For simplicity, use transpose + matmul directly
    let b_t = transpose(b);
    let a_t = transpose(a);
    let grad_a = batch_matmul(grad_output, &b_t);
    let grad_b = batch_matmul(&a_t, grad_output);
    vec![grad_a, grad_b]
}

// ============================================================
// Helper: transpose (reuse)
// ============================================================

fn transpose<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let shape = a.shape();
    assert!(shape.len() >= 2, "transpose requires at least 2D tensor");

    let mut new_shape = shape.to_vec();
    let last = new_shape.len() - 1;
    new_shape.swap(last - 1, last);

    let rows = shape[last - 1];
    let cols = shape[last];
    let batch_stride = rows * cols;

    let data: Vec<T> = a
        .data()
        .par_chunks(batch_stride)
        .flat_map(|chunk| {
            let mut transposed = vec![T::from_f32(0.0); batch_stride];
            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = chunk[i * cols + j];
                }
            }
            transposed
        })
        .collect();

    Tensor::new(data, &new_shape)
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct BatchMatMulOp;

impl<T: DType + Send + Sync> Operator<T> for BatchMatMulOp {
    fn name(&self) -> &'static str {
        "batch_matmul"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        batch_matmul(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        batch_matmul_backward(grad, inputs[0], inputs[1])
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_matmul_2x2x2() {
        let a = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        );
        let b = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            &[2, 2, 2],
        );
        let c = batch_matmul(&a, &b);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
        assert_eq!(c.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_batch_matmul_broadcast() {
        // a: [2, 2, 3], b: [1, 3, 2] → broadcast b to [2, 3, 2]
        let a = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[2, 2, 3],
        );
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
        let c = batch_matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2, 2]);
        // Manual verification for first batch:
        // [[1,2,3],[4,5,6]] * [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
        // Second batch:
        // [[7,8,9],[10,11,12]] * [[1,2],[3,4],[5,6]] = [[76,100],[103,136]]
        assert_eq!(c.data()[0], 22.0);
        assert_eq!(c.data()[1], 28.0);
        assert_eq!(c.data()[2], 49.0);
        assert_eq!(c.data()[3], 64.0);
        assert_eq!(c.data()[4], 76.0);
        assert_eq!(c.data()[5], 100.0);
        assert_eq!(c.data()[6], 103.0);
        assert_eq!(c.data()[7], 136.0);
    }

    #[test]
    fn test_batch_matmul_backward() {
        let grad = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            &[2, 2, 2],
        );
        let a = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        );
        let b = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            &[2, 2, 2],
        );
        let grads = batch_matmul_backward(&grad, &a, &b);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 2, 2]);
        assert_eq!(grads[1].shape(), &[2, 2, 2]);
    }
}
