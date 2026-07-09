// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Max Forward
// ============================================================

pub fn max<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    let shape = a.shape();
    assert!(dim < shape.len(), "max: dim out of range");

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
            let mut max_val = f32::NEG_INFINITY;
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                let v = a_data[idx].to_f32();
                if v > max_val {
                    max_val = v;
                }
            }
            let out_idx = o * inner + i;
            out_data[out_idx] = T::from_f32(max_val);
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
// Min Forward
// ============================================================

pub fn min<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    let shape = a.shape();
    assert!(dim < shape.len(), "min: dim out of range");

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
            let mut min_val = f32::INFINITY;
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                let v = a_data[idx].to_f32();
                if v < min_val {
                    min_val = v;
                }
            }
            let out_idx = o * inner + i;
            out_data[out_idx] = T::from_f32(min_val);
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
// Backward - 简化版   TODO: 完善
// ============================================================

pub fn max_backward<T: DType>(
    grad_output: &Tensor<T>,
    _a: &Tensor<T>,
    _dim: usize,
    _keepdim: bool,
) -> Vec<Tensor<T>> {
    // 简化版：直接传递梯度
    vec![grad_output.clone()]
}

pub fn min_backward<T: DType>(
    grad_output: &Tensor<T>,
    _a: &Tensor<T>,
    _dim: usize,
    _keepdim: bool,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait 实现
// ============================================================

pub struct MaxOp;

impl<T: DType + Send + Sync> Operator<T> for MaxOp {
    fn name(&self) -> &'static str {
        "max"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        max(inputs[0], dim, keepdim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        max_backward(grad, inputs[0], dim, keepdim)
    }
}

pub struct MinOp;

impl<T: DType + Send + Sync> Operator<T> for MinOp {
    fn name(&self) -> &'static str {
        "min"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        min(inputs[0], dim, keepdim)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        min_backward(grad, inputs[0], dim, keepdim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_1d() {
        let a = Tensor::new(vec![1.0, 5.0, 3.0, 4.0], &[4]);
        let c = max(&a, 0, false);
        assert_eq!(c.data(), &[5.0]);
    }

    #[test]
    fn test_max_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = max(&a, 1, false);
        assert_eq!(c.data(), &[3.0, 6.0]);
    }

    #[test]
    fn test_min_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = min(&a, 1, false);
        assert_eq!(c.data(), &[1.0, 4.0]);
    }
}
