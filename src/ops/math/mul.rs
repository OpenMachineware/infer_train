use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn mul<T: DType + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in mul");

    let data: Vec<T> = a
        .data()
        .par_iter()
        .zip(b.data().par_iter())
        .map(|(&x, &y)| x * y)
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn mul_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for i in 0..grad_a.len() {
        grad_a.data_mut()[i] = grad_a.data()[i] * b.data()[i];
        grad_b.data_mut()[i] = grad_b.data()[i] * a.data()[i];
    }
    vec![grad_a, grad_b]
}

// ============================================================
// 3. 量化 Forward
// ============================================================

pub fn quantized_mul(a: &Tensor<i8>, b: &Tensor<i8>) -> Tensor<i8> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in quantized_mul");

    let scale_a = a.scale().unwrap_or(1.0);
    let zero_a = a.zero_point().unwrap_or(0.0);
    let scale_b = b.scale().unwrap_or(1.0);
    let zero_b = b.zero_point().unwrap_or(0.0);

    let scale = scale_a;
    let zero = zero_a;

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| {
            let x_fp = (x as f32 - zero_a) * scale_a;
            let y_fp = (y as f32 - zero_b) * scale_b;
            x_fp * y_fp
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// 4. 量化 Backward
// ============================================================

pub fn quantized_mul_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let mut grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for i in 0..grad_a.len() {
        grad_a.data_mut()[i] = grad_a.data()[i].saturating_mul(b.data()[i]);
        grad_b.data_mut()[i] = grad_b.data()[i].saturating_mul(a.data()[i]);
    }
    vec![grad_a, grad_b]
}

// ============================================================
// 5. Operator Trait 实现
// ============================================================

pub struct MulOp;

impl<T: DType + Send + Sync> Operator<T> for MulOp {
    fn name(&self) -> &'static str {
        "mul"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        mul(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        mul_backward(grad, inputs[0], inputs[1])
    }
}

pub struct QuantizedMulOp;

impl Operator<i8> for QuantizedMulOp {
    fn name(&self) -> &'static str {
        "quantized_mul"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 2);
        quantized_mul(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 2);
        quantized_mul_backward(grad, inputs[0], inputs[1])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_f32() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], &[3]);
        let c = mul(&a, &b);
        assert_eq!(c.data(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_mul_backward() {
        let grad = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], &[3]);
        let grads = mul_backward(&grad, &a, &b);
        assert_eq!(grads[0].data(), &[4.0, 10.0, 18.0]);
        assert_eq!(grads[1].data(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_quantized_mul() {
        let a = Tensor::<i8>::new_quantized(vec![10, 20, 30], &[3], 0.1, 0.0);
        let b = Tensor::<i8>::new_quantized(vec![4, 5, 6], &[3], 0.1, 0.0);
        let c = quantized_mul(&a, &b);
        assert_eq!(c.data(), &[4, 10, 18]);
    }
}
