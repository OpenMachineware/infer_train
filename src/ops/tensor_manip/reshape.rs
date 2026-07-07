// src/ops/tensor_manip/reshape.rs

use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 泛型 Forward
// ============================================================

pub fn reshape<T: DType + Send + Sync>(input: &Tensor<T>, new_shape: &[usize]) -> Tensor<T> {
    let total: usize = input.shape().iter().product();
    let new_total: usize = new_shape.iter().product();
    assert_eq!(total, new_total, "reshape: total elements must match");
    Tensor::new(input.data().to_vec(), new_shape)
}

// ============================================================
// 泛型 Backward
// ============================================================

pub fn reshape_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
) -> Vec<Tensor<T>> {
    vec![reshape(grad_output, original_shape)]
}

// ============================================================
// 量化 Forward
// ============================================================

pub fn quantized_reshape(input: &Tensor<i8>, new_shape: &[usize]) -> Tensor<i8> {
    let total: usize = input.shape().iter().product();
    let new_total: usize = new_shape.iter().product();
    assert_eq!(total, new_total, "quantized_reshape: total elements must match");
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(input.data().to_vec(), new_shape, scale, zero_point)
}

// ============================================================
// 量化 Backward
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
    fn name(&self) -> &'static str { "reshape" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let shape = attrs.get_int_list("shape")
            .expect("reshape requires shape")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        reshape(inputs[0], &shape)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        reshape_backward(grad, inputs[0].shape())
    }
}

pub struct QuantizedReshapeOp;

impl Operator<i8> for QuantizedReshapeOp {
    fn name(&self) -> &'static str { "quantized_reshape" }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let shape = attrs.get_int_list("shape")
            .expect("quantized_reshape requires shape")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        quantized_reshape(inputs[0], &shape)
    }
    fn backward(&self, grad: &Tensor<i8>, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Vec<Tensor<i8>> {
        quantized_reshape_backward(grad, inputs[0].shape())
    }
    fn supports_quantized(&self) -> bool { true }
}
