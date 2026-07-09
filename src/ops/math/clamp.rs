use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 浮点泛型 Forward
// ============================================================

pub fn clamp<T: DType + Send + Sync>(
    a: &Tensor<T>,
    min: f32,
    max: f32,
) -> Tensor<T> {
    let data: Vec<T> = a
        .data()
        .par_iter()
        .map(|&x| {
            let v = x.to_f32();
            T::from_f32(v.clamp(min, max))
        })
        .collect();

    Tensor::new(data, a.shape())
}

// ============================================================
// 浮点泛型 Backward
// ============================================================

pub fn clamp_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    min: f32,
    max: f32,
) -> Vec<Tensor<T>> {
    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = a.data()[i].to_f32();
        grad.data_mut()[i] = if v >= min && v <= max {
            grad.data()[i]
        } else {
            T::from_f32(0.0)
        };
    }
    vec![grad]
}

// ============================================================
// 量化 Forward
// ============================================================

pub fn quantized_clamp(a: &Tensor<i8>, min: f32, max: f32) -> Tensor<i8> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let result_fp: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let v = (x as f32 - zero) * scale;
            v.clamp(min, max)
        })
        .collect();

    let data: Vec<i8> = result_fp
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, a.shape(), scale, zero)
}

// ============================================================
// 量化 Backward
// ============================================================

pub fn quantized_clamp_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    min: f32,
    max: f32,
) -> Vec<Tensor<i8>> {
    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let mut grad = grad_output.clone();
    for i in 0..grad.len() {
        let v = (a.data()[i] as f32 - zero) * scale;
        grad.data_mut()[i] =
            if v >= min && v <= max { grad.data()[i] } else { 0 };
    }
    vec![grad]
}

// ============================================================
// Operator Trait 实现
// ============================================================

pub struct ClampOp;

impl<T: DType + Send + Sync> Operator<T> for ClampOp {
    fn name(&self) -> &'static str {
        "clamp"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let min = attrs.get_float("min").unwrap_or(0.0);
        let max = attrs.get_float("max").unwrap_or(1.0);
        clamp(inputs[0], min, max)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        let min = attrs.get_float("min").unwrap_or(0.0);
        let max = attrs.get_float("max").unwrap_or(1.0);
        clamp_backward(grad, inputs[0], min, max)
    }
}

pub struct QuantizedClampOp;

impl Operator<i8> for QuantizedClampOp {
    fn name(&self) -> &'static str {
        "quantized_clamp"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let min = attrs.get_float("min").unwrap_or(0.0);
        let max = attrs.get_float("max").unwrap_or(1.0);
        quantized_clamp(inputs[0], min, max)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        let min = attrs.get_float("min").unwrap_or(0.0);
        let max = attrs.get_float("max").unwrap_or(1.0);
        quantized_clamp_backward(grad, inputs[0], min, max)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_f32() {
        let a = Tensor::new(vec![-1.0, 0.5, 2.0, 3.5], &[4]);
        let c = clamp(&a, 0.0, 2.0);
        assert_eq!(c.data(), &[0.0, 0.5, 2.0, 2.0]);
    }
}
