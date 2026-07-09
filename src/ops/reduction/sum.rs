// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// 浮点泛型 Forward
// ============================================================

pub fn sum<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    let shape = a.shape();
    assert!(dim < shape.len(), "sum: dim out of range");

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
            let mut sum_val = T::from_f32(0.0);
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                sum_val = sum_val + a_data[idx];
            }
            let out_idx = o * inner + i;
            out_data[out_idx] = sum_val;
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
// 浮点泛型 Backward
// ============================================================

pub fn sum_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    dim: usize,
    _keepdim: bool,
) -> Vec<Tensor<T>> {
    let shape = a.shape();
    let dim_size = shape[dim];

    let mut grad = vec![T::from_f32(0.0); a.len()];
    let grad_data = grad_output.data();

    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let stride = dim_size * inner;

    for o in 0..outer {
        for i in 0..inner {
            let grad_idx = o * inner + i;
            let g = grad_data[grad_idx];
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                grad[idx] = g;
            }
        }
    }

    vec![Tensor::new(grad, shape)]
}

// ============================================================
// Operator Trait 实现
// ============================================================

pub struct SumOp;

impl<T: DType + Send + Sync> Operator<T> for SumOp {
    fn name(&self) -> &'static str {
        "sum"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        sum(inputs[0], dim, keepdim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        sum_backward(grad, inputs[0], dim, keepdim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_1d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let c = sum(&a, 0, false);
        assert_eq!(c.data(), &[10.0]);
        let empty_shape: &[usize] = &[];
        assert_eq!(c.shape(), empty_shape);
    }

    #[test]
    fn test_sum_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = sum(&a, 1, false);
        assert_eq!(c.data(), &[6.0, 15.0]);
        assert_eq!(c.shape(), &[2]);
    }

    #[test]
    fn test_sum_keepdim() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = sum(&a, 0, true);
        assert_eq!(c.shape(), &[1, 2]);
        assert_eq!(c.data(), &[4.0, 6.0]);
    }
}
