use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn relu<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a.data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            T::from_f32(if v > 0.0 { v } else { 0.0 })
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn relu_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        grad.data_mut()[i] = if v > 0.0 {
            grad.data()[i]
        } else {
            T::from_f32(0.0)
        };
    }
    vec![grad]
}

// ============================================================
// 3. 量化 Forward
// ============================================================

pub fn quantized_relu(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = a.data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            let q = if v > 0.0 { x } else { zero as i8 };
            q.clamp(-128, 127)
        })
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// 4. 量化 Backward
// ============================================================

pub fn quantized_relu_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let zero = a.zero_point().unwrap_or(0.0) as i8;
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        grad.data_mut()[i] = if a.data()[i] > zero {
            grad.data()[i]
        } else {
            0
        };
    }
    vec![grad]
}

// ============================================================
// 5. Operator Trait 实现
// ============================================================

pub struct ReluOp;

impl<T: DType + Send + Sync> Operator<T> for ReluOp {
    fn name(&self) -> &'static str { "relu" }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        relu(inputs[0])
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        relu_backward(grad, inputs[0])
    }
}

pub struct QuantizedReluOp;

impl Operator<i8> for QuantizedReluOp {
    fn name(&self) -> &'static str { "quantized_relu" }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_relu(inputs[0])
    }
    fn backward(&self, grad: &Tensor<i8>, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_relu_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_f32() {
        let a = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], &[4]);
        let c = relu(&a);
        assert_eq!(c.data(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_relu_backward() {
        let grad = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let a = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], &[4]);
        let grads = relu_backward(&grad, &a);
        assert_eq!(grads[0].data(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_quantized_relu() {
        let a = Tensor::<i8>::new_quantized(vec![-10, 20, -30, 40], &[4], 0.1, 0.0);
        let c = quantized_relu(&a);
        assert_eq!(c.data(), &[0, 20, 0, 40]);
    }
}
