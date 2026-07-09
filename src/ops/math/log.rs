use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn log<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            if v > 0.0 {
                T::from_f32(v.ln())
            } else {
                T::from_f32(0.0)
            }
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn log_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // ∂L/∂a = ∂L/∂output / a
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        if v > 0.0 {
            grad.data_mut()[i] = T::from_f32(grad.data()[i].to_f32() / v);
        } else {
            grad.data_mut()[i] = T::from_f32(0.0);
        }
    }
    vec![grad]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_log(a: &Tensor<i8>) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            if v > 0.0 {
                v.ln()
            } else {
                0.0
            }
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

pub fn quantized_log_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        if a.data()[i] > 0 {
            grad.data_mut()[i] = grad.data()[i] / a.data()[i];
        } else {
            grad.data_mut()[i] = 0;
        }
    }
    vec![grad]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct LogOp;

impl<T: DType + Send + Sync> Operator<T> for LogOp {
    fn name(&self) -> &'static str {
        "log"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        log(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        log_backward(grad, inputs[0])
    }
}

pub struct QuantizedLogOp;

impl Operator<i8> for QuantizedLogOp {
    fn name(&self) -> &'static str {
        "quantized_log"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_log(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        quantized_log_backward(grad, inputs[0])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_f32() {
        let a = Tensor::new(vec![1.0, 2.718281, 7.389056], &[3]);
        let c = log(&a);
        let eps = 1e-6;
        assert!(f32::abs(c.data()[0] - 0.0) < eps);
        assert!(f32::abs(c.data()[1] - 1.0) < eps);
        assert!(f32::abs(c.data()[2] - 2.0) < eps);
    }
}
