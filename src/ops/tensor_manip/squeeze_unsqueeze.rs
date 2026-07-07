// src/ops/tensor_manip/squeeze_unsqueeze.rs

use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// Squeeze 泛型
// ============================================================

pub fn squeeze<T: DType + Send + Sync>(input: &Tensor<T>, dim: Option<usize>) -> Tensor<T> {
    let shape = input.shape();
    let mut new_shape = Vec::new();

    if let Some(d) = dim {
        assert!(d < shape.len(), "squeeze: dim out of range");
        assert_eq!(shape[d], 1, "squeeze: dim size must be 1");
        for i in 0..shape.len() {
            if i != d {
                new_shape.push(shape[i]);
            }
        }
    } else {
        for &d in shape {
            if d != 1 {
                new_shape.push(d);
            }
        }
    }

    Tensor::new(input.data().to_vec(), &new_shape)
}

pub fn squeeze_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
    dim: Option<usize>,
) -> Vec<Tensor<T>> {
    if let Some(d) = dim {
        let mut new_shape = original_shape.to_vec();
        new_shape.insert(d, 1);
        vec![Tensor::new(grad_output.data().to_vec(), &new_shape)]
    } else {
        vec![Tensor::new(grad_output.data().to_vec(), original_shape)]
    }
}

// ============================================================
// Unsqueeze 泛型
// ============================================================

pub fn unsqueeze<T: DType + Send + Sync>(input: &Tensor<T>, dim: usize) -> Tensor<T> {
    let shape = input.shape();
    let mut new_shape = Vec::with_capacity(shape.len() + 1);
    for i in 0..=dim {
        if i < shape.len() {
            new_shape.push(shape[i]);
        } else {
            break;
        }
    }
    new_shape.push(1);
    for i in dim..shape.len() {
        new_shape.push(shape[i]);
    }
    Tensor::new(input.data().to_vec(), &new_shape)
}

pub fn unsqueeze_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_dim: usize,
) -> Vec<Tensor<T>> {
    let shape = grad_output.shape();
    let mut new_shape = Vec::new();
    for i in 0..shape.len() {
        if i != original_dim {
            new_shape.push(shape[i]);
        }
    }
    vec![Tensor::new(grad_output.data().to_vec(), &new_shape)]
}

// ============================================================
// 量化 Squeeze Forward
// ============================================================

pub fn quantized_squeeze(input: &Tensor<i8>, dim: Option<usize>) -> Tensor<i8> {
    let shape = input.shape();
    let mut new_shape = Vec::new();

    if let Some(d) = dim {
        assert!(d < shape.len(), "quantized_squeeze: dim out of range");
        assert_eq!(shape[d], 1, "quantized_squeeze: dim size must be 1");
        for i in 0..shape.len() {
            if i != d {
                new_shape.push(shape[i]);
            }
        }
    } else {
        for &d in shape {
            if d != 1 {
                new_shape.push(d);
            }
        }
    }

    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(input.data().to_vec(), &new_shape, scale, zero_point)
}

// ============================================================
// 量化 Squeeze Backward
// ============================================================

pub fn quantized_squeeze_backward(
    grad_output: &Tensor<i8>,
    original_shape: &[usize],
    dim: Option<usize>,
) -> Vec<Tensor<i8>> {
    let scale = grad_output.scale().unwrap_or(1.0);
    let zero_point = grad_output.zero_point().unwrap_or(0.0);

    if let Some(d) = dim {
        let mut new_shape = original_shape.to_vec();
        new_shape.insert(d, 1);
        vec![Tensor::<i8>::new_quantized(grad_output.data().to_vec(), &new_shape, scale, zero_point)]
    } else {
        vec![Tensor::<i8>::new_quantized(grad_output.data().to_vec(), original_shape, scale, zero_point)]
    }
}

// ============================================================
// 量化 Unsqueeze Forward
// ============================================================

pub fn quantized_unsqueeze(input: &Tensor<i8>, dim: usize) -> Tensor<i8> {
    let shape = input.shape();
    let mut new_shape = Vec::with_capacity(shape.len() + 1);
    for i in 0..=dim {
        if i < shape.len() {
            new_shape.push(shape[i]);
        } else {
            break;
        }
    }
    new_shape.push(1);
    for i in dim..shape.len() {
        new_shape.push(shape[i]);
    }
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(input.data().to_vec(), &new_shape, scale, zero_point)
}

// ============================================================
// 量化 Unsqueeze Backward
// ============================================================

pub fn quantized_unsqueeze_backward(
    grad_output: &Tensor<i8>,
    original_dim: usize,
) -> Vec<Tensor<i8>> {
    let scale = grad_output.scale().unwrap_or(1.0);
    let zero_point = grad_output.zero_point().unwrap_or(0.0);
    let shape = grad_output.shape();
    let mut new_shape = Vec::new();
    for i in 0..shape.len() {
        if i != original_dim {
            new_shape.push(shape[i]);
        }
    }
    vec![Tensor::<i8>::new_quantized(grad_output.data().to_vec(), &new_shape, scale, zero_point)]
}

// ============================================================
// Operator Trait
// ============================================================

pub struct SqueezeOp;

impl<T: DType + Send + Sync> Operator<T> for SqueezeOp {
    fn name(&self) -> &'static str { "squeeze" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").map(|d| d as usize);
        squeeze(inputs[0], dim)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        let dim = attrs.get_int("dim").map(|d| d as usize);
        squeeze_backward(grad, inputs[0].shape(), dim)
    }
}

pub struct UnsqueezeOp;

impl<T: DType + Send + Sync> Operator<T> for UnsqueezeOp {
    fn name(&self) -> &'static str { "unsqueeze" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        unsqueeze(inputs[0], dim)
    }
    fn backward(&self, grad: &Tensor<T>, _inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        unsqueeze_backward(grad, dim)
    }
}

// ============================================================
// 量化 Operator Trait
// ============================================================

pub struct QuantizedSqueezeOp;

impl Operator<i8> for QuantizedSqueezeOp {
    fn name(&self) -> &'static str { "quantized_squeeze" }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").map(|d| d as usize);
        quantized_squeeze(inputs[0], dim)
    }
    fn backward(&self, grad: &Tensor<i8>, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Vec<Tensor<i8>> {
        let dim = attrs.get_int("dim").map(|d| d as usize);
        quantized_squeeze_backward(grad, inputs[0].shape(), dim)
    }
    fn supports_quantized(&self) -> bool { true }
}

pub struct QuantizedUnsqueezeOp;

impl Operator<i8> for QuantizedUnsqueezeOp {
    fn name(&self) -> &'static str { "quantized_unsqueeze" }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        quantized_unsqueeze(inputs[0], dim)
    }
    fn backward(&self, grad: &Tensor<i8>, _inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Vec<Tensor<i8>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        quantized_unsqueeze_backward(grad, dim)
    }
    fn supports_quantized(&self) -> bool { true }
}
