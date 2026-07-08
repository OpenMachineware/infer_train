use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn tanh<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            T::from_f32(v.tanh())
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn tanh_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        let tanh_v = v.tanh();
        grad.data_mut()[i] =
            T::from_f32(grad.data()[i].to_f32() * (1.0 - tanh_v * tanh_v));
    }
    vec![grad]
}

// ============================================================
// 3. 量化 Forward (简化版)
// ============================================================

pub fn quantized_tanh(a: &Tensor<i8>) -> Tensor<i8> {
    let a_fp = a.dequantize().expect("Failed to dequantize");
    let c_fp = tanh(&a_fp);

    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = c_fp
        .data()
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, c_fp.shape(), scale, zero)
}

// ============================================================
// 4. 量化 Backward (简化版)
// ============================================================

pub fn quantized_tanh_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let grad_fp = grad_output.dequantize().expect("Failed to dequantize grad");
    let a_fp = a.dequantize().expect("Failed to dequantize a");
    let grads = tanh_backward(&grad_fp, &a_fp);

    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = grads[0]
        .data()
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    vec![Tensor::<i8>::new_quantized(data, grads[0].shape(), scale, zero)]
}

// ============================================================
// 5. Operator Trait 实现
// ============================================================

pub struct TanhOp;

impl<T: DType + Send + Sync> Operator<T> for TanhOp {
    fn name(&self) -> &'static str {
        "tanh"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        tanh(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        tanh_backward(grad, inputs[0])
    }
}

pub struct QuantizedTanhOp;

impl Operator<i8> for QuantizedTanhOp {
    fn name(&self) -> &'static str {
        "quantized_tanh"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_tanh(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_tanh_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_f32() {
        let a = Tensor::new(vec![-1.0, 0.0, 1.0], &[3]);
        let c = tanh(&a);
        let eps = 0.01;
        assert!(f32::abs(c.data()[0] + 0.76159) < eps);
        assert!(f32::abs(c.data()[1] - 0.0) < eps);
        assert!(f32::abs(c.data()[2] - 0.76159) < eps);
    }
}
