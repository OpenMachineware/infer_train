use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn leaky_relu<T: DType + Send + Sync>(
    a: &Tensor<T>,
    alpha: f32,
) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            T::from_f32(if v > 0.0 { v } else { v * alpha })
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn leaky_relu_backward<T: DType>(
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
            T::from_f32(grad.data()[i].to_f32() * alpha)
        };
    }
    vec![grad]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_leaky_relu(a: &Tensor<i8>, alpha: f32) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = a
        .data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            let q = if v > 0.0 {
                x
            } else {
                ((v * alpha / scale) + zero).round() as i8
            };
            q.clamp(-128, 127)
        })
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// Quantized Backward
// ============================================================

pub fn quantized_leaky_relu_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    alpha: f32,
) -> Vec<Tensor<i8>> {
    let zero = a.zero_point().unwrap_or(0.0) as i8;
    let _alpha_i8 = (alpha * 128.0) as i8;
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        grad.data_mut()[i] = if a.data()[i] > zero {
            grad.data()[i]
        } else {
            // grad.data()[i].saturating_mul(alpha_i8) / 128
            // FIXME Simplified for now, to be improved later
            grad.data()[i]
        };
    }
    vec![grad]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct LeakyReluOp;

impl<T: DType + Send + Sync> Operator<T> for LeakyReluOp {
    fn name(&self) -> &'static str {
        "leaky_relu"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(0.01);
        leaky_relu(inputs[0], alpha)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(0.01);
        leaky_relu_backward(grad, inputs[0], alpha)
    }
}

pub struct QuantizedLeakyReluOp;

impl Operator<i8> for QuantizedLeakyReluOp {
    fn name(&self) -> &'static str {
        "quantized_leaky_relu"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(0.01);
        quantized_leaky_relu(inputs[0], alpha)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        let alpha = attrs.get_float("alpha").unwrap_or(0.01);
        quantized_leaky_relu_backward(grad, inputs[0], alpha)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // src/ops/activation/leaky_relu.rs

    #[test]
    fn test_leaky_relu_f32() {
        let a = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], &[4]);
        let c = leaky_relu(&a, 0.01);
        let eps = 0.001;
        assert!(f32::abs(c.data()[0] - (-0.01)) < eps);
        assert!(f32::abs(c.data()[1] - 2.0) < eps);
        assert!(f32::abs(c.data()[2] - (-0.03)) < eps);
        assert!(f32::abs(c.data()[3] - 4.0) < eps);
    }
}
