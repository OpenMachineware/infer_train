// src/ops/tensor_manip/concat.rs

use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 泛型 Forward
// ============================================================

pub fn concat<T: DType + Send + Sync>(inputs: &[&Tensor<T>], dim: usize) -> Tensor<T> {
    assert!(!inputs.is_empty(), "concat: at least one input required");

    let first_shape = inputs[0].shape();
    let rank = first_shape.len();
    assert!(dim < rank, "concat: dim out of range");

    let mut out_shape = first_shape.to_vec();
    let mut total_dim = 0;
    let mut total_len = 0;

    for input in inputs {
        assert_eq!(input.shape().len(), rank, "concat: all inputs must have same rank");
        for (i, &d) in input.shape().iter().enumerate() {
            if i != dim {
                assert_eq!(d, first_shape[i], "concat: shape mismatch at dim {}", i);
            }
        }
        total_dim += input.shape()[dim];
        total_len += input.len();
    }
    out_shape[dim] = total_dim;

    let mut all_data = Vec::with_capacity(total_len);
    for input in inputs {
        all_data.extend_from_slice(input.data());
    }

    Tensor::new(all_data, &out_shape)
}

// ============================================================
// 泛型 Backward
// ============================================================

pub fn concat_backward<T: DType>(
    grad_output: &Tensor<T>,
    input_shapes: &[&[usize]],
    _dim: usize,
) -> Vec<Tensor<T>> {
    let mut result = Vec::new();
    let mut offset = 0;
    for &shape in input_shapes {
        let len: usize = shape.iter().product();
        let grad_slice = grad_output.data()[offset..offset + len].to_vec();
        result.push(Tensor::new(grad_slice, shape));
        offset += len;
    }
    result
}

// ============================================================
// 量化 Forward
// ============================================================

pub fn quantized_concat(inputs: &[&Tensor<i8>], dim: usize) -> Tensor<i8> {
    assert!(!inputs.is_empty(), "quantized_concat: at least one input required");

    let first_shape = inputs[0].shape();
    let rank = first_shape.len();
    assert!(dim < rank, "quantized_concat: dim out of range");

    let scale = inputs[0].scale().unwrap_or(1.0);
    let zero_point = inputs[0].zero_point().unwrap_or(0.0);

    let mut out_shape = first_shape.to_vec();
    let mut total_dim = 0;
    let mut total_len = 0;

    for input in inputs {
        assert_eq!(input.shape().len(), rank, "quantized_concat: all inputs must have same rank");
        for (i, &d) in input.shape().iter().enumerate() {
            if i != dim {
                assert_eq!(d, first_shape[i], "quantized_concat: shape mismatch at dim {}", i);
            }
        }
        total_dim += input.shape()[dim];
        total_len += input.len();
    }
    out_shape[dim] = total_dim;

    let mut all_data = Vec::with_capacity(total_len);
    for input in inputs {
        all_data.extend_from_slice(input.data());
    }

    Tensor::<i8>::new_quantized(all_data, &out_shape, scale, zero_point)
}

// ============================================================
// 量化 Backward
// ============================================================

pub fn quantized_concat_backward(
    grad_output: &Tensor<i8>,
    input_shapes: &[&[usize]],
    _dim: usize,
) -> Vec<Tensor<i8>> {
    let scale = grad_output.scale().unwrap_or(1.0);
    let zero_point = grad_output.zero_point().unwrap_or(0.0);
    let mut result = Vec::new();
    let mut offset = 0;
    for &shape in input_shapes {
        let len: usize = shape.iter().product();
        let grad_slice = grad_output.data()[offset..offset + len].to_vec();
        result.push(Tensor::<i8>::new_quantized(grad_slice, shape, scale, zero_point));
        offset += len;
    }
    result
}

// ============================================================
// Operator Trait
// ============================================================

pub struct ConcatOp;

impl<T: DType + Send + Sync> Operator<T> for ConcatOp {
    fn name(&self) -> &'static str { "concat" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        concat(inputs, dim)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape()).collect();
        concat_backward(grad, &shapes, dim)
    }
}

pub struct QuantizedConcatOp;

impl Operator<i8> for QuantizedConcatOp {
    fn name(&self) -> &'static str { "quantized_concat" }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        quantized_concat(inputs, dim)
    }
    fn backward(&self, grad: &Tensor<i8>, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Vec<Tensor<i8>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape()).collect();
        quantized_concat_backward(grad, &shapes, dim)
    }
    fn supports_quantized(&self) -> bool { true }
}
