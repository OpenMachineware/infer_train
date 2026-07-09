use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn sub<T: DType + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in sub");

    let data: Vec<T> = a
        .data()
        .par_iter()
        .zip(b.data().par_iter())
        .map(|(&x, &y)| x - y)
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn sub_backward<T: DType>(
    grad_output: &Tensor<T>,
    _a: &Tensor<T>,
    _b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for v in grad_b.data_mut() {
        *v = -*v;
    }
    vec![grad_a, grad_b]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_sub(a: &Tensor<i8>, b: &Tensor<i8>) -> Tensor<i8> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in quantized_sub");

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
            x_fp - y_fp
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// Quantized Backward
// ============================================================

pub fn quantized_sub_backward(
    grad_output: &Tensor<i8>,
    _a: &Tensor<i8>,
    _b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for v in grad_b.data_mut() {
        *v = -*v;
    }
    vec![grad_a, grad_b]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct SubOp;

impl<T: DType + Send + Sync> Operator<T> for SubOp {
    fn name(&self) -> &'static str {
        "sub"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        sub(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        sub_backward(grad, inputs[0], inputs[1])
    }
}

pub struct QuantizedSubOp;

impl Operator<i8> for QuantizedSubOp {
    fn name(&self) -> &'static str {
        "quantized_sub"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 2);
        quantized_sub(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 2);
        quantized_sub_backward(grad, inputs[0], inputs[1])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_f32() {
        let a = Tensor::new(vec![5.0, 7.0, 9.0], &[3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], &[3]);
        let c = sub(&a, &b);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sub_backward() {
        let grad = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let a = Tensor::new(vec![1.0, 1.0, 1.0], &[3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], &[3]);
        let grads = sub_backward(&grad, &a, &b);
        assert_eq!(grads[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(grads[1].data(), &[-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_quantized_sub() {
        let a = Tensor::<i8>::new_quantized(vec![5, 7, 9], &[3], 0.1, 0.0);
        let b = Tensor::<i8>::new_quantized(vec![4, 5, 6], &[3], 0.1, 0.0);
        let c = quantized_sub(&a, &b);
        assert_eq!(c.data(), &[1, 2, 3]);
    }
}
