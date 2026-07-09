use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 浮点泛型 Forward
// ============================================================

pub fn relu6<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            T::from_f32(if v > 0.0 {
                if v < 6.0 {
                    v
                } else {
                    6.0
                }
            } else {
                0.0
            })
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// 浮点泛型 Backward
// ============================================================

pub fn relu6_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        grad.data_mut()[i] =
            if v > 0.0 && v < 6.0 { grad.data()[i] } else { T::from_f32(0.0) };
    }
    vec![grad]
}

// ============================================================
// 量化 Forward
// ============================================================

// src/ops/activation/relu6.rs

pub fn quantized_relu6(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);
    let six = (6.0 / scale) as i8;
    let zero_i8 = zero as i8;

    let data: Vec<i8> = a
        .data()
        .iter()
        .map(|&x| {
            if x < zero_i8 {
                zero_i8
            } else if x > zero_i8 + six {
                zero_i8 + six
            } else {
                x
            }
        })
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// 量化 Backward
// ============================================================

pub fn quantized_relu6_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);
    let six = (6.0 / scale) as i8;
    let zero_i8 = zero as i8;

    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i];
        grad.data_mut()[i] =
            if v > zero_i8 && v < zero_i8 + six { grad.data()[i] } else { 0 };
    }
    vec![grad]
}

// ============================================================
// Operator Trait 实现
// ============================================================

pub struct Relu6Op;

impl<T: DType + Send + Sync> Operator<T> for Relu6Op {
    fn name(&self) -> &'static str {
        "relu6"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        relu6(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        relu6_backward(grad, inputs[0])
    }
}

pub struct QuantizedRelu6Op;

impl Operator<i8> for QuantizedRelu6Op {
    fn name(&self) -> &'static str {
        "quantized_relu6"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_relu6(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_relu6_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu6_f32() {
        let a = Tensor::new(vec![-1.0, 2.0, 6.0, 7.0, 3.0], &[5]);
        let c = relu6(&a);
        assert_eq!(c.data(), &[0.0, 2.0, 6.0, 6.0, 3.0]);
    }
}
