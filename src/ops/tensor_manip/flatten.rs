// src/ops/tensor_manip/flatten.rs

use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Generic Forward
// ============================================================

pub fn flatten<T: DType + Send + Sync>(
    input: &Tensor<T>,
    start_dim: usize,
    end_dim: usize,
) -> Tensor<T> {
    let shape = input.shape();
    let rank = shape.len();
    let start = start_dim.min(rank);
    let end = end_dim.min(rank);

    let mut new_shape = Vec::new();
    for i in 0..start {
        new_shape.push(shape[i]);
    }
    let mut flattened = 1;
    for i in start..=end {
        flattened *= shape[i];
    }
    new_shape.push(flattened);
    for i in end + 1..rank {
        new_shape.push(shape[i]);
    }

    Tensor::new(input.data().to_vec(), &new_shape)
}

// ============================================================
// Generic Backward
// ============================================================

pub fn flatten_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
) -> Vec<Tensor<T>> {
    vec![Tensor::new(grad_output.data().to_vec(), original_shape)]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_flatten(
    input: &Tensor<i8>,
    start_dim: usize,
    end_dim: usize,
) -> Tensor<i8> {
    let shape = input.shape();
    let rank = shape.len();
    let start = start_dim.min(rank);
    let end = end_dim.min(rank);

    let mut new_shape = Vec::new();
    for i in 0..start {
        new_shape.push(shape[i]);
    }
    let mut flattened = 1;
    for i in start..=end {
        flattened *= shape[i];
    }
    new_shape.push(flattened);
    for i in end + 1..rank {
        new_shape.push(shape[i]);
    }

    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(
        input.data().to_vec(),
        &new_shape,
        scale,
        zero_point,
    )
}

// ============================================================
// Quantized Backward
// ============================================================

pub fn quantized_flatten_backward(
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
// Flatten Operator (Generic)
// ============================================================

pub struct FlattenOp;

impl<T: DType + Send + Sync> Operator<T> for FlattenOp {
    fn name(&self) -> &'static str {
        "flatten"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let start_dim = attrs.get_int("start_dim").unwrap_or(0) as usize;
        // If end_dim is not provided, use rank - 1
        let end_dim =
            attrs.get_int("end_dim").map(|v| v as usize).unwrap_or_else(|| {
                let rank = inputs[0].shape().len();
                if rank > 0 {
                    rank - 1
                } else {
                    0
                }
            });
        flatten(inputs[0], start_dim, end_dim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        flatten_backward(grad, inputs[0].shape())
    }
}

// ============================================================
// Quantized Flatten Operator
// ============================================================

pub struct QuantizedFlattenOp;

impl Operator<i8> for QuantizedFlattenOp {
    fn name(&self) -> &'static str {
        "quantized_flatten"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let start_dim = attrs.get_int("start_dim").unwrap_or(0) as usize;
        // If end_dim is not provided, use rank - 1
        let end_dim =
            attrs.get_int("end_dim").map(|v| v as usize).unwrap_or_else(|| {
                let rank = inputs[0].shape().len();
                if rank > 0 {
                    rank - 1
                } else {
                    0
                }
            });
        quantized_flatten(inputs[0], start_dim, end_dim)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        quantized_flatten_backward(grad, inputs[0].shape())
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}
