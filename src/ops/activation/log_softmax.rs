use super::softmax::softmax;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn log_softmax<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
) -> Tensor<T> {
    let sm = softmax(a, dim);
    let data: Vec<T> =
        sm.data().par_iter().map(|&x| T::from_f32(x.to_f32().ln())).collect();

    Tensor::new(data, sm.shape())
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn log_softmax_backward<T: DType>(
    grad_output: &Tensor<T>,
    output: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // log_softmax backward: ∂L/∂x = grad - sum(grad) * exp(log_softmax)
    let mut grad = grad_output.clone();
    let out_data = output.data();
    let _grad_data = grad.data_mut();

    // 简化版
    for i in 0..grad.len() {
        let grad_val = grad.data()[i].to_f32();
        let out_val = out_data[i].to_f32();
        grad.data_mut()[i] = T::from_f32(grad_val * out_val.exp());
    }
    vec![grad]
}

// ============================================================
// 3. 量化 Forward
// ============================================================

pub fn quantized_log_softmax(a: &Tensor<i8>, dim: usize) -> Tensor<i8> {
    let a_fp = a.dequantize().expect("Failed to dequantize");
    let c_fp = log_softmax(&a_fp, dim);

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
// 4. 量化 Backward (简化版)
// ============================================================

pub fn quantized_log_softmax_backward(
    grad_output: &Tensor<i8>,
    output: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let grad_fp = grad_output.dequantize().expect("Failed to dequantize grad");
    let out_fp = output.dequantize().expect("Failed to dequantize output");
    let grads = log_softmax_backward(&grad_fp, &out_fp);

    let scale = output.scale().unwrap_or(1.0);
    let zero = output.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = grads[0]
        .data()
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    vec![Tensor::<i8>::new_quantized(data, grads[0].shape(), scale, zero)]
}

// ============================================================
// 5. Operator Trait 实现
// ============================================================

pub struct LogSoftmaxOp;

impl<T: DType + Send + Sync> Operator<T> for LogSoftmaxOp {
    fn name(&self) -> &'static str {
        "log_softmax"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        log_softmax(inputs[0], actual_dim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        let output = log_softmax(inputs[0], actual_dim);
        log_softmax_backward(grad, &output)
    }
}

pub struct QuantizedLogSoftmaxOp;

impl Operator<i8> for QuantizedLogSoftmaxOp {
    fn name(&self) -> &'static str {
        "quantized_log_softmax"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        quantized_log_softmax(inputs[0], actual_dim)
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        inputs: &[&Tensor<i8>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(-1) as usize;
        let actual_dim =
            if dim == usize::MAX { inputs[0].shape().len() - 1 } else { dim };
        let output = quantized_log_softmax(inputs[0], actual_dim);
        quantized_log_softmax_backward(grad, &output)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax_f32() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = log_softmax(&a, 0);
        assert!(c.data()[0] < -2.0);
        assert!(c.data()[1] < -1.0);
        assert!(c.data()[2] > -1.0);
    }
}
