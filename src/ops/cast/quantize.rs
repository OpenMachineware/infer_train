use rayon::prelude::*;
// use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// 1. Quantize (量化: f32 -> i8)
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
// 2. Dequantize (反量化: i8 -> f32)
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
// 3. Quantize Backward (梯度传递: 量化不可导，但可以用直通估计器)
// ============================================================

pub fn quantize_backward(grad_output: &Tensor<f32>) -> Vec<Tensor<f32>> {
    // 直通估计器 (STE): 梯度直接传递
    vec![grad_output.clone()]
}

// ============================================================
// 4. Operators
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
        // 返回 f32 张量 (实际量化后是 i8，但为了兼容性)
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
        // 假设输入是量化后的 f32 表示，实际是 i8
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
        // -10 在 [-128, 127] 范围内，保持 -10
        assert_eq!(quantized.data()[0], -10);
        assert_eq!(quantized.data()[1], 0);
        assert_eq!(quantized.data()[2], 10);
    }

    #[test]
    fn test_quantize_clamp() {
        let input = Tensor::new(vec![-1000.0, 1000.0], &[2]);
        let quantized = quantize(&input, 1.0, 0.0);
        // -1000 被 clamp 到 -128, 1000 被 clamp 到 127
        assert_eq!(quantized.data()[0], -128);
        assert_eq!(quantized.data()[1], 127);
    }
}
