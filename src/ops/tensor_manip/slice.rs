// src/ops/tensor_manip/slice.rs

use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 泛型 Forward
// ============================================================

pub fn slice<T: DType + Send + Sync>(
    input: &Tensor<T>,
    dim: usize,
    start: i64,
    end: i64,
    step: i64,
) -> Tensor<T> {
    let shape = input.shape();
    let dim_size = shape[dim] as i64;
    let start = if start < 0 { dim_size + start } else { start };
    let end = if end < 0 { dim_size + end } else { end.min(dim_size) };
    let step = step.abs();

    let out_len = ((end - start) / step) as usize;
    let mut out_shape = shape.to_vec();
    out_shape[dim] = out_len;

    // 计算步长
    let inner_stride = shape[dim + 1..].iter().product::<usize>();
    let outer = shape[..dim].iter().product::<usize>();
    let dim_stride = inner_stride * shape[dim];

    let mut data = Vec::with_capacity(outer * out_len * inner_stride);
    for o in 0..outer {
        for i in 0..out_len {
            let idx = o * dim_stride + (start + i as i64 * step) as usize * inner_stride;
            data.extend_from_slice(&input.data()[idx..idx + inner_stride]);
        }
    }

    Tensor::new(data, &out_shape)
}

// ============================================================
// 泛型 Backward
// ============================================================

pub fn slice_backward<T: DType>(
    grad_output: &Tensor<T>,
    original_shape: &[usize],
    dim: usize,
    start: i64,
    end: i64,
    step: i64,
) -> Vec<Tensor<T>> {
    let mut data = vec![T::from_f32(0.0); original_shape.iter().product()];
    // 把梯度填回对应的位置
    let inner_stride = original_shape[dim + 1..].iter().product::<usize>();
    let outer = original_shape[..dim].iter().product::<usize>();
    let dim_size = original_shape[dim] as i64;
    let start = if start < 0 { dim_size + start } else { start };
    let _end = if end < 0 { dim_size + end } else { end.min(dim_size) };
    let step = step.abs();

    for o in 0..outer {
        for i in 0..grad_output.shape()[dim] {
            let src_idx = (o * grad_output.shape()[dim] + i) * inner_stride;
            let dst_idx = o * dim_size as usize * inner_stride + (start + i as i64 * step) as usize * inner_stride;
            for j in 0..inner_stride {
                data[dst_idx + j] = grad_output.data()[src_idx + j];
            }
        }
    }

    vec![Tensor::new(data, original_shape)]
}

// ============================================================
// 量化 Forward
// ============================================================

pub fn quantized_slice(
    input: &Tensor<i8>,
    dim: usize,
    start: i64,
    end: i64,
    step: i64,
) -> Tensor<i8> {
    let shape = input.shape();
    let dim_size = shape[dim] as i64;
    let start = if start < 0 { dim_size + start } else { start };
    let end = if end < 0 { dim_size + end } else { end.min(dim_size) };
    let step = step.abs();

    let out_len = ((end - start) / step) as usize;
    let mut out_shape = shape.to_vec();
    out_shape[dim] = out_len;

    let inner_stride = shape[dim + 1..].iter().product::<usize>();
    let outer = shape[..dim].iter().product::<usize>();
    let dim_stride = inner_stride * shape[dim];

    let mut data = Vec::with_capacity(outer * out_len * inner_stride);
    for o in 0..outer {
        for i in 0..out_len {
            let idx = o * dim_stride + (start + i as i64 * step) as usize * inner_stride;
            data.extend_from_slice(&input.data()[idx..idx + inner_stride]);
        }
    }

    let scale = input.scale().unwrap_or(1.0);
    let zero_point = input.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(data, &out_shape, scale, zero_point)
}

// ============================================================
// 量化 Backward
// ============================================================

pub fn quantized_slice_backward(
    grad_output: &Tensor<i8>,
    original_shape: &[usize],
    dim: usize,
    start: i64,
    end: i64,
    step: i64,
) -> Vec<Tensor<i8>> {
    let mut data = vec![0i8; original_shape.iter().product()];
    let inner_stride = original_shape[dim + 1..].iter().product::<usize>();
    let outer = original_shape[..dim].iter().product::<usize>();
    let dim_size = original_shape[dim] as i64;
    let start = if start < 0 { dim_size + start } else { start };
    let _end = if end < 0 { dim_size + end } else { end.min(dim_size) };
    let step = step.abs();

    let scale = grad_output.scale().unwrap_or(1.0);
    let zero_point = grad_output.zero_point().unwrap_or(0.0);

    for o in 0..outer {
        for i in 0..grad_output.shape()[dim] {
            let src_idx = (o * grad_output.shape()[dim] + i) * inner_stride;
            let dst_idx = o * dim_size as usize * inner_stride + (start + i as i64 * step) as usize * inner_stride;
            for j in 0..inner_stride {
                data[dst_idx + j] = grad_output.data()[src_idx + j];
            }
        }
    }

    vec![Tensor::<i8>::new_quantized(data, original_shape, scale, zero_point)]
}

// ============================================================
// Operator Trait
// ============================================================

pub struct SliceOp;

impl<T: DType + Send + Sync> Operator<T> for SliceOp {
    fn name(&self) -> &'static str { "slice" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let start = attrs.get_int("start").unwrap_or(0);
        let end = attrs.get_int("end").unwrap_or(i64::MAX);
        let step = attrs.get_int("step").unwrap_or(1);
        slice(inputs[0], dim, start, end, step)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let start = attrs.get_int("start").unwrap_or(0);
        let end = attrs.get_int("end").unwrap_or(i64::MAX);
        let step = attrs.get_int("step").unwrap_or(1);
        slice_backward(grad, inputs[0].shape(), dim, start, end, step)
    }
}

pub struct QuantizedSliceOp;

impl Operator<i8> for QuantizedSliceOp {
    fn name(&self) -> &'static str { "quantized_slice" }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let start = attrs.get_int("start").unwrap_or(0);
        let end = attrs.get_int("end").unwrap_or(i64::MAX);
        let step = attrs.get_int("step").unwrap_or(1);
        quantized_slice(inputs[0], dim, start, end, step)
    }
    fn backward(&self, grad: &Tensor<i8>, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Vec<Tensor<i8>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let start = attrs.get_int("start").unwrap_or(0);
        let end = attrs.get_int("end").unwrap_or(i64::MAX);
        let step = attrs.get_int("step").unwrap_or(1);
        quantized_slice_backward(grad, inputs[0].shape(), dim, start, end, step)
    }
    fn supports_quantized(&self) -> bool { true }
}
