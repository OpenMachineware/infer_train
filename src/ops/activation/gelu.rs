use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// Float Generic Forward - Approximate Version
// ============================================================

pub fn gelu<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            // GELU approximation:
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let c = 0.79788456; // sqrt(2/pi)
            let x3 = v * v * v;
            let tanh_val = (c * (v + 0.044715 * x3)).tanh();
            T::from_f32(0.5 * v * (1.0 + tanh_val))
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn gelu_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        let c = 0.79788456;
        let x3 = v * v * v;
        let tanh_val = (c * (v + 0.044715 * x3)).tanh();
        let cosh_val = (c * (v + 0.044715 * x3)).cosh();
        let sech2 = 1.0 / (cosh_val * cosh_val);

        let dgelu = 0.5 * (1.0 + tanh_val)
            + 0.5 * v * sech2 * (c * (1.0 + 3.0 * 0.044715 * v * v));
        grad.data_mut()[i] = T::from_f32(grad.data()[i].to_f32() * dgelu);
    }
    vec![grad]
}

// ============================================================
// Quantized Forward - Simplified TODO: Improve
// ============================================================

pub fn quantized_gelu(a: &Tensor<i8>) -> Tensor<i8> {
    let a_fp = a.dequantize().expect("Failed to dequantize");
    let c_fp = gelu(&a_fp);

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
// Quantized Backward - Simplified TODO: Improve
// ============================================================

pub fn quantized_gelu_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let grad_fp = grad_output.dequantize().expect("Failed to dequantize grad");
    let a_fp = a.dequantize().expect("Failed to dequantize a");
    let grads = gelu_backward(&grad_fp, &a_fp);

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

pub struct GeluOp;

impl<T: DType + Send + Sync> Operator<T> for GeluOp {
    fn name(&self) -> &'static str {
        "gelu"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        gelu(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        gelu_backward(grad, inputs[0])
    }
}

pub struct QuantizedGeluOp;

impl Operator<i8> for QuantizedGeluOp {
    fn name(&self) -> &'static str {
        "quantized_gelu"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_gelu(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_gelu_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_f32() {
        let a = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        let c = gelu(&a);
        let eps = 0.01;
        assert!(f32::abs(c.data()[0] + 0.1587) < eps);
        assert!(f32::abs(c.data()[1] - 0.0) < eps);
        assert!(f32::abs(c.data()[2] - 0.8413) < eps);
        assert!(f32::abs(c.data()[3] - 1.9546) < eps);
    }
}
