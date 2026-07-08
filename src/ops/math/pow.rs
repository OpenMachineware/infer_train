use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn pow<T: DType + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in pow");

    let data: Vec<T> = a
        .data()
        .par_iter()
        .zip(b.data().par_iter())
        .map(|(&x, &y)| T::from_f32(x.to_f32().powf(y.to_f32())))
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn pow_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    let mut grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for i in 0..grad_a.len() {
        let a_f32 = a.data()[i].to_f32();
        let b_f32 = b.data()[i].to_f32();
        if a_f32 > 0.0 {
            grad_a.data_mut()[i] = T::from_f32(
                grad_a.data()[i].to_f32() * b_f32 * a_f32.powf(b_f32 - 1.0),
            );
            grad_b.data_mut()[i] = T::from_f32(
                grad_b.data()[i].to_f32() * a_f32.powf(b_f32) * a_f32.ln(),
            );
        } else {
            grad_a.data_mut()[i] = T::from_f32(0.0);
            grad_b.data_mut()[i] = T::from_f32(0.0);
        }
    }
    vec![grad_a, grad_b]
}

// ============================================================
// 3. 量化 Forward
// ============================================================

pub fn quantized_pow(a: &Tensor<i8>, b: &Tensor<i8>) -> Tensor<i8> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in quantized_pow");

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
            x_fp.powf(y_fp)
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

pub fn quantized_pow_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    // 简化的量化反向传播
    let mut grad_a = grad_output.clone();
    let mut grad_b = grad_output.clone();
    for i in 0..grad_a.len() {
        let a_val = a.data()[i];
        let b_val = b.data()[i];
        if a_val > 0 {
            grad_a.data_mut()[i] = grad_a.data()[i].saturating_mul(b_val);
            grad_b.data_mut()[i] = grad_b.data()[i].saturating_mul(a_val);
        } else {
            grad_a.data_mut()[i] = 0;
            grad_b.data_mut()[i] = 0;
        }
    }
    vec![grad_a, grad_b]
}

// ============================================================
// 5. Operator Trait 实现
// ============================================================

pub struct PowOp;

impl<T: DType + Send + Sync> Operator<T> for PowOp {
    fn name(&self) -> &'static str {
        "pow"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        pow(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        pow_backward(grad, inputs[0], inputs[1])
    }
}

pub struct QuantizedPowOp;

impl Operator<i8> for QuantizedPowOp {
    fn name(&self) -> &'static str {
        "quantized_pow"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 2);
        quantized_pow(inputs[0], inputs[1])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 2);
        quantized_pow_backward(grad, inputs[0], inputs[1])
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_f32() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0], &[3]);
        let b = Tensor::new(vec![2.0, 2.0, 2.0], &[3]);
        let c = pow(&a, &b);
        assert_eq!(c.data(), &[4.0, 9.0, 16.0]);
    }
}
