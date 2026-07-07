// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 辅助：获取形状
// ============================================================

fn get_matmul_dims(a: &[usize], b: &[usize]) -> (usize, usize, usize, usize, usize, usize) {
    // 返回: (batch_dims, m, k, n, a_rank, b_rank)
    let a_rank = a.len();
    let b_rank = b.len();

    let m = a[a_rank - 2];
    let k_a = a[a_rank - 1];
    let k_b = b[b_rank - 2];
    let n = b[b_rank - 1];

    assert_eq!(k_a, k_b, "Matmul: inner dimensions must match: {} vs {}", k_a, k_b);

    let batch_dims = a_rank.max(b_rank) - 2;
    (batch_dims, m, k_a, n, a_rank, b_rank)
}

fn get_batch_stride(shape: &[usize], dim: usize) -> usize {
    if dim >= shape.len() {
        return 1;
    }
    shape[dim..].iter().product()
}

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn matmul<T: DType + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (batch_dims, m, k, n, a_rank, b_rank) = get_matmul_dims(a_shape, b_shape);

    // 计算输出形状
    let mut out_shape = Vec::new();
    for i in 0..batch_dims {
        let a_idx = if i < a_rank - 2 { a_shape[i] } else { 1 };
        let b_idx = if i < b_rank - 2 { b_shape[i] } else { 1 };
        out_shape.push(a_idx.max(b_idx));
    }
    out_shape.push(m);
    out_shape.push(n);

    let out_size: usize = out_shape.iter().product();
    let mut out_data = vec![T::from_f32(0.0); out_size];

    let _a_batch_stride = if a_rank >= 2 { get_batch_stride(a_shape, a_rank - 2) } else { 1 };
    let _b_batch_stride = if b_rank >= 2 { get_batch_stride(b_shape, b_rank - 2) } else { 1 };
    let out_batch_stride = m * n;

    let batch_total: usize = out_shape[..batch_dims].iter().product();
    let a_data = a.data();
    let b_data = b.data();

    // 预计算每个 batch 的偏移
    let mut batch_offsets = Vec::with_capacity(batch_total);
    for batch_idx in 0..batch_total {
        let mut a_offset = 0;
        let mut b_offset = 0;
        let mut tmp = batch_idx;
        for dim in (0..batch_dims).rev() {
            let dim_size = out_shape[dim];
            let idx = tmp % dim_size;
            tmp /= dim_size;

            if dim < a_rank - 2 {
                let a_dim = a_shape[dim];
                let a_idx = if a_dim == 1 { 0 } else { idx % a_dim };
                a_offset += a_idx * a_shape[dim + 1..].iter().product::<usize>();
            }
            if dim < b_rank - 2 {
                let b_dim = b_shape[dim];
                let b_idx = if b_dim == 1 { 0 } else { idx % b_dim };
                b_offset += b_idx * b_shape[dim + 1..].iter().product::<usize>();
            }
        }
        batch_offsets.push((a_offset, b_offset));
    }

    for batch_idx in 0..batch_total {
        let (a_offset, b_offset) = batch_offsets[batch_idx];
        let out_offset = batch_idx * out_batch_stride;
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::from_f32(0.0);
                for t in 0..k {
                    let a_idx = a_offset + i * k + t;
                    let b_idx = b_offset + t * n + j;
                    sum = sum + a_data[a_idx] * b_data[b_idx];
                }
                out_data[out_offset + i * n + j] = sum;
            }
        }
    }

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn matmul_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Vec<Tensor<T>> {
    // 使用 crate::ops::linalg::transpose
    let b_t = crate::ops::linalg::transpose::transpose(b);
    let a_t = crate::ops::linalg::transpose::transpose(a);
    let grad_a = matmul(grad_output, &b_t);
    let grad_b = matmul(&a_t, grad_output);
    vec![grad_a, grad_b]
}

// ============================================================
// 3. 量化 Forward (简化版)
// ============================================================

pub fn quantized_matmul(a: &Tensor<i8>, b: &Tensor<i8>) -> Tensor<i8> {
    let a_fp = a.dequantize().expect("Failed to dequantize a");
    let b_fp = b.dequantize().expect("Failed to dequantize b");
    let c_fp = matmul(&a_fp, &b_fp);

    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let data: Vec<i8> = c_fp.data()
        .iter()
        .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Tensor::<i8>::new_quantized(data, c_fp.shape(), scale, zero)
}

// ============================================================
// 4. 量化 Backward
// ============================================================

pub fn quantized_matmul_backward(
    grad_output: &Tensor<i8>,
    a: &Tensor<i8>,
    b: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    let grad_fp = grad_output.dequantize().expect("Failed to dequantize grad");
    let a_fp = a.dequantize().expect("Failed to dequantize a");
    let b_fp = b.dequantize().expect("Failed to dequantize b");

    let grads = matmul_backward(&grad_fp, &a_fp, &b_fp);

    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);

    let mut result = Vec::new();
    for grad in grads {
        let data: Vec<i8> = grad.data()
            .iter()
            .map(|&v| ((v / scale) + zero).round().clamp(-128.0, 127.0) as i8)
            .collect();
        result.push(Tensor::<i8>::new_quantized(data, grad.shape(), scale, zero));
    }
    result
}

// ============================================================
// 5. Operator Trait 实现
// ============================================================

pub struct MatMulOp;

impl<T: DType + Send + Sync> Operator<T> for MatMulOp {
    fn name(&self) -> &'static str { "matmul" }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        matmul(inputs[0], inputs[1])
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        matmul_backward(grad, inputs[0], inputs[1])
    }
}

pub struct QuantizedMatMulOp;

impl Operator<i8> for QuantizedMatMulOp {
    fn name(&self) -> &'static str { "quantized_matmul" }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 2);
        quantized_matmul(inputs[0], inputs[1])
    }
    fn backward(&self, grad: &Tensor<i8>, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Vec<Tensor<i8>> {
        assert_eq!(inputs.len(), 2);
        quantized_matmul_backward(grad, inputs[0], inputs[1])
    }
    fn supports_quantized(&self) -> bool { true }
}

// ============================================================
// 6. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::new(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.data(), &[4.0, 6.0, 10.0, 12.0]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_matmul_batch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
        assert_eq!(c.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_matmul_backward() {
        let grad = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::new(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
        let grads = matmul_backward(&grad, &a, &b);
        assert_eq!(grads[0].shape(), &[2, 2]);
        assert_eq!(grads[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_quantized_matmul() {
        let a = Tensor::<i8>::new_quantized(vec![10, 20, 30, 40], &[2, 2], 0.1, 0.0);
        let b = Tensor::<i8>::new_quantized(vec![20, 0, 10, 30], &[2, 2], 0.1, 0.0);
        let c = quantized_matmul(&a, &b);
        assert_eq!(c.data(), &[40, 60, 100, 120]);
        assert_eq!(c.shape(), &[2, 2]);
    }
}
