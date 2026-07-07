// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. TopK Forward
// ============================================================

pub fn topk<T: DType + Send + Sync>(
    input: &Tensor<T>,
    k: usize,
    dim: usize,
    largest: bool,
) -> (Tensor<T>, Tensor<i64>) {
    let shape = input.shape();
    assert!(dim < shape.len(), "topk: dim out of range");

    let dim_size = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let stride = dim_size * inner;

    let input_data = input.data();

    let mut out_shape = shape.to_vec();
    out_shape[dim] = k;
    let out_size: usize = out_shape.iter().product();

    let mut out_values = vec![T::from_f32(0.0); out_size];
    let mut out_indices = vec![0i64; out_size];

    for o in 0..outer {
        for i in 0..inner {
            // 收集该维度的所有值
            let mut pairs: Vec<(f32, usize)> = Vec::with_capacity(dim_size);
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                pairs.push((input_data[idx].to_f32(), d));
            }

            // 排序
            if largest {
                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            } else {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            }

            // 取前 k 个
            let out_base = (o * k + 0) * inner + i;
            for d in 0..k.min(dim_size) {
                let out_idx = out_base + d * inner;
                out_values[out_idx] = T::from_f32(pairs[d].0);
                out_indices[out_idx] = pairs[d].1 as i64;
            }
        }
    }

    let out_shape_indices = out_shape.clone();
    (Tensor::new(out_values, &out_shape), Tensor::new(out_indices, &out_shape_indices))
}

// ============================================================
// 2. TopK Backward (简化版)
// ============================================================

pub fn topk_backward<T: DType>(
    grad_output: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct TopkOp;

impl<T: DType + Send + Sync> Operator<T> for TopkOp {
    fn name(&self) -> &'static str { "topk" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let k = attrs.get_int("k").unwrap_or(1) as usize;
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let largest = attrs.get_bool("largest").unwrap_or(true);
        let (values, _indices) = topk(inputs[0], k, dim, largest);
        values
    }
    fn backward(&self, grad: &Tensor<T>, _inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        topk_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_1d() {
        let input = Tensor::new(vec![5.0, 2.0, 8.0, 1.0, 9.0], &[5]);
        let (values, indices) = topk(&input, 3, 0, true);
        assert_eq!(values.data(), &[9.0, 8.0, 5.0]);
        assert_eq!(indices.data(), &[4, 2, 0]);
    }

    #[test]
    fn test_topk_smallest() {
        let input = Tensor::new(vec![5.0, 2.0, 8.0, 1.0, 9.0], &[5]);
        let (values, indices) = topk(&input, 3, 0, false);
        assert_eq!(values.data(), &[1.0, 2.0, 5.0]);
        assert_eq!(indices.data(), &[3, 1, 0]);
    }

    #[test]
    fn test_topk_2d() {
        let input = Tensor::new(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], &[3, 3]);
        let (values, indices) = topk(&input, 2, 1, true);
        assert_eq!(values.shape(), &[3, 2]);
        assert_eq!(indices.shape(), &[3, 2]);
        assert_eq!(values.data()[0], 3.0);
        assert_eq!(values.data()[1], 2.0);
        assert_eq!(values.data()[2], 6.0);
        assert_eq!(values.data()[3], 5.0);
    }
}
