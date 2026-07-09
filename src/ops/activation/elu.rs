use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn elu<T: DType + Send + Sync>(a: &Tensor<T>, alpha: f32) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            T::from_f32(if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn elu_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    alpha: f32,
) -> Vec<Tensor<T>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        grad.data_mut()[i] = if v > 0.0 {
            grad.data()[i]
        } else {
            T::from_f32(grad.data()[i].to_f32() * alpha * v.exp())
        };
    }
    vec![grad]
}

// ============================================================
// Quantized Forward (Simplified)
// ============================================================

pub fn quantized_elu(a: &Tensor<i8>, alpha: f32) -> Tensor<i8> {
    let a_fp = a.dequantize().expect("Failed to dequantize");
    let c_fp = elu(&a_fp, alpha);

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
// Quantized Backward (Simplified)
// ============================================================

pub fn quantized_elu_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    alpha: f32,
) -> Vec<Tensor<i8>> {
    let grad_fp = grad_output.dequantize().expect("Failed to dequantize grad");
    let a_fp = a.dequantize().expect("Failed to dequantize a");
    let grads = elu_backward(&grad_fp, &a_fp, alpha);

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
// Operator Trait Implementation
// ============================================================

pub struct EluOp;

impl<T: DType + Send + Sync> Operator<T> for EluOp {
    fn name(&self) -> &'static str {
        "elu"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(1.0);
        elu(inputs[0], alpha)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(1.0);
        elu_backward(grad, inputs[0], alpha)
    }
}

pub struct QuantizedEluOp;

impl Operator<i8> for QuantizedEluOp {
    fn name(&self) -> &'static str {
        "quantized_elu"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(1.0);
        quantized_elu(inputs[0], alpha)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(1.0);
        quantized_elu_backward(grad, inputs[0], alpha)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elu_f32() {
        let a = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        let c = elu(&a, 1.0);
        let eps = 0.001;
        assert!(f32::abs(c.data()[0] + 0.6321) < eps);
        assert!(f32::abs(c.data()[1] - 0.0) < eps);
        assert!(f32::abs(c.data()[2] - 1.0) < eps);
        assert!(f32::abs(c.data()[3] - 2.0) < eps);
    }
}
