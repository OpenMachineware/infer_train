use rayon::prelude::*;
// use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Quantize (Quantization: f32 -> i8)
// ============================================================

pub fn quantize(
    input: &Tensor<f32>,
    scale: f32,
    zero_point: f32,
) -> Tensor<i8> {
    let data: Vec<i8> = input
        .data()
        .par_iter()
        .map(|&x| {
            let v = (x / scale) + zero_point;
            v.round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    Tensor::<i8>::new_quantized(data, input.shape(), scale, zero_point)
}

// ============================================================
// Dequantize (Dequantization: i8 -> f32)
// ============================================================

pub fn dequantize(input: &Tensor<i8>) -> Tensor<f32> {
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);

    let data: Vec<f32> = input
        .data()
        .par_iter()
        .map(|&x| (x as f32 - zero_point) * scale)
        .collect();

    Tensor::new(data, input.shape())
}

// ============================================================
// Quantize Backward (Gradient Propagation: quantization is non-differentiable,
//                    but we can use Straight-Through Estimator)
// ============================================================

pub fn quantize_backward(grad_output: &Tensor<f32>) -> Vec<Tensor<f32>> {
    // Straight-Through Estimator (STE): gradient passes through directly
    vec![grad_output.clone()]
}

// ============================================================
// Operators
// ============================================================

pub struct QuantizeOp;

impl Operator<f32> for QuantizeOp {
    fn name(&self) -> &'static str {
        "quantize"
    }
    fn forward(&self, inputs: &[&Tensor<f32>], attrs: &OpAttrs) -> Tensor<f32> {
        assert_eq!(inputs.len(), 1);
        let scale = attrs.get_float("scale").unwrap_or(1.0);
        let zero_point = attrs.get_float("zero_point").unwrap_or(0.0);
        let quantized = quantize(inputs[0], scale, zero_point);
        // Return f32 tensor
        // (actually i8 after quantization, but for compatibility)
        dequantize(&quantized)
    }
    fn backward(
        &self,
        grad: &Tensor<f32>,
        _inputs: &[&Tensor<f32>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<f32>> {
        quantize_backward(grad)
    }
}

pub struct DequantizeOp;

impl Operator<f32> for DequantizeOp {
    fn name(&self) -> &'static str {
        "dequantize"
    }
    fn forward(
        &self,
        inputs: &[&Tensor<f32>],
        _attrs: &OpAttrs,
    ) -> Tensor<f32> {
        assert_eq!(inputs.len(), 1);
        // Assume input is quantized f32 representation, actually i8
        inputs[0].clone()
    }
    fn backward(
        &self,
        grad: &Tensor<f32>,
        _inputs: &[&Tensor<f32>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<f32>> {
        vec![grad.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let input = Tensor::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5], &[6]);
        let quantized = quantize(&input, 0.1, 0.0);
        assert_eq!(quantized.data(), &[0, 1, 2, 3, 4, 5]);

        let dequantized = dequantize(&quantized);
        assert_eq!(dequantized.data(), &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn test_quantize_range() {
        let input = Tensor::new(vec![-10.0, 0.0, 10.0], &[3]);
        let quantized = quantize(&input, 1.0, 0.0);
        // -10 is within [-128, 127] range, stays -10
        assert_eq!(quantized.data()[0], -10);
        assert_eq!(quantized.data()[1], 0);
        assert_eq!(quantized.data()[2], 10);
    }

    #[test]
    fn test_quantize_clamp() {
        let input = Tensor::new(vec![-1000.0, 1000.0], &[2]);
        let quantized = quantize(&input, 1.0, 0.0);
        // -1000 clamped to -128, 1000 clamped to 127
        assert_eq!(quantized.data()[0], -128);
        assert_eq!(quantized.data()[1], 127);
    }
}
