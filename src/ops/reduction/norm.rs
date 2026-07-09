// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// 浮点泛型 Forward
// ============================================================

pub fn norm<T: DType + Send + Sync>(
    a: &Tensor<T>,
    p: usize,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    let shape = a.shape();
    assert!(dim < shape.len(), "norm: dim out of range");

    let dim_size = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let stride = dim_size * inner;

    let a_data = a.data();
    let out_size = if keepdim {
        let mut new_shape = shape.to_vec();
        new_shape[dim] = 1;
        new_shape.iter().product()
    } else {
        let mut new_shape = shape.to_vec();
        new_shape.remove(dim);
        new_shape.iter().product()
    };

    let mut out_data = vec![T::from_f32(0.0); out_size];

    for o in 0..outer {
        for i in 0..inner {
            let mut sum = 0.0;
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                let v = a_data[idx].to_f32();
                sum += v.abs().powf(p as f32);
            }
            let out_idx = o * inner + i;
            out_data[out_idx] = T::from_f32(sum.powf(1.0 / p as f32));
        }
    }

    let out_shape = if keepdim {
        let mut new_shape = shape.to_vec();
        new_shape[dim] = 1;
        new_shape
    } else {
        let mut new_shape = shape.to_vec();
        new_shape.remove(dim);
        new_shape
    };

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// 浮点泛型 Backward - 简化版   TODO: 完善
// ============================================================

pub fn norm_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait 实现
// ============================================================

pub struct NormOp;

impl<T: DType + Send + Sync> Operator<T> for NormOp {
    fn name(&self) -> &'static str {
        "norm"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let p = attrs.get_int("p").unwrap_or(2) as usize;
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        norm(inputs[0], p, dim, keepdim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        norm_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_l2_1d() {
        let a = Tensor::new(vec![3.0, 4.0], &[2]);
        let c = norm(&a, 2, 0, false);
        let eps = 0.001;
        assert!(f32::abs(c.data()[0] - 5.0) < eps);
    }

    #[test]
    fn test_norm_l1_2d() {
        let a = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]);
        let c = norm(&a, 1, 1, false);
        // 第一行: |-1| + |2| = 3, 第二行: |-3| + |4| = 7
        assert_eq!(c.data(), &[3.0, 7.0]);
    }
}
