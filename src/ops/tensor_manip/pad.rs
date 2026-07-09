// src/ops/tensor_manip/pad.rs

use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Generic Forward
// ============================================================

pub fn pad<T: DType + Send + Sync>(
    input: &Tensor<T>,
    pad: &[usize],
    value: f32,
) -> Tensor<T> {
    let shape = input.shape();
    assert_eq!(pad.len(), shape.len() * 2, "pad: pad length must be 2 * rank");

    let mut new_shape = Vec::new();
    for i in 0..shape.len() {
        new_shape.push(shape[i] + pad[2 * i] + pad[2 * i + 1]);
    }

    let mut data = vec![T::from_f32(value); new_shape.iter().product()];
    // Simplified: copy data to the middle position
    let _inner_stride = shape[shape.len() - 1];
    let outer = shape[..shape.len() - 1].iter().product::<usize>();
    let input_data = input.data();

    for o in 0..outer {
        for i in 0..shape[shape.len() - 1] {
            let src_idx = o * shape[shape.len() - 1] + i;
            let dst_idx = (o + pad[0]) * new_shape[shape.len() - 1]
                + (i + pad[pad.len() - 2]);
            data[dst_idx] = input_data[src_idx];
        }
    }
    Tensor::new(data, &new_shape)
}

// ============================================================
// Generic Backward
// ============================================================

pub fn pad_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
    pad: &[usize],
) -> Vec<Tensor<T>> {
    let mut data = Vec::new();
    let shape = grad_output.shape();
    let inner_stride = shape[shape.len() - 1];
    let outer = shape[..shape.len() - 1].iter().product::<usize>();
    let grad_data = grad_output.data();

    for o in 0..outer {
        for i in 0..original_shape[original_shape.len() - 1] {
            let src_idx =
                (o + pad[0]) * inner_stride + (i + pad[pad.len() - 2]);
            data.push(grad_data[src_idx]);
        }
    }
    vec![Tensor::new(data, original_shape)]
}

// ============================================================
// Quantized Forward
// ============================================================

pub fn quantized_pad(
    input: &Tensor<i8>,
    pad: &[usize],
    value: f32,
) -> Tensor<i8> {
    let shape = input.shape();
    assert_eq!(
        pad.len(),
        shape.len() * 2,
        "quantized_pad: pad length must be 2 * rank"
    );

    let mut new_shape = Vec::new();
    for i in 0..shape.len() {
        new_shape.push(shape[i] + pad[2 * i] + pad[2 * i + 1]);
    }

    let mut data = vec![value as i8; new_shape.iter().product()];
    let _inner_stride = shape[shape.len() - 1];
    let outer = shape[..shape.len() - 1].iter().product::<usize>();
    let input_data = input.data();

    for o in 0..outer {
        for i in 0..shape[shape.len() - 1] {
            let src_idx = o * shape[shape.len() - 1] + i;
            let dst_idx = (o + pad[0]) * new_shape[shape.len() - 1]
                + (i + pad[pad.len() - 2]);
            data[dst_idx] = input_data[src_idx];
        }
    }
    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(data, &new_shape, scale, zero_point)
}

// ============================================================
// Quantized Backward
// ============================================================

pub fn quantized_pad_backward(
    grad_output: &Tensor<i8>,
    original_shape: &[usize],
    pad: &[usize],
) -> Vec<Tensor<i8>> {
    let mut data = Vec::new();
    let shape = grad_output.shape();
    let inner_stride = shape[shape.len() - 1];
    let outer = shape[..shape.len() - 1].iter().product::<usize>();
    let grad_data = grad_output.data();
    let scale = grad_output.scale().unwrap_or(1.0);
    let zero_point = grad_output.zero_point().unwrap_or(0.0);

    for o in 0..outer {
        for i in 0..original_shape[original_shape.len() - 1] {
            let src_idx =
                (o + pad[0]) * inner_stride + (i + pad[pad.len() - 2]);
            data.push(grad_data[src_idx]);
        }
    }
    vec![Tensor::<i8>::new_quantized(data, original_shape, scale, zero_point)]
}

// ============================================================
// Pad Operator
// ============================================================

pub struct PadOp;

impl<T: DType + Send + Sync> Operator<T> for PadOp {
    fn name(&self) -> &'static str {
        "pad"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let padding = attrs
            .get_int_list("pad")
            .expect("pad requires pad")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        let value = attrs.get_float("value").unwrap_or(0.0);
        pad(inputs[0], &padding, value)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        let padding = attrs
            .get_int_list("pad")
            .expect("pad requires pad")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        pad_backward(grad, inputs[0].shape(), &padding)
    }
}

// ============================================================
// Quantized Pad Operator
// ============================================================

pub struct QuantizedPadOp;

impl Operator<i8> for QuantizedPadOp {
    fn name(&self) -> &'static str {
        "quantized_pad"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let padding = attrs
            .get_int_list("pad")
            .expect("quantized_pad requires pad")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        let value = attrs.get_float("value").unwrap_or(0.0);
        quantized_pad(inputs[0], &padding, value)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        let padding = attrs
            .get_int_list("pad")
            .expect("quantized_pad requires pad")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        quantized_pad_backward(grad, inputs[0].shape(), &padding)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}
