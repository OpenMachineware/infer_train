// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::OpAttrs;
use crate::tensor::Tensor;

// ============================================================
// 1. ArgMax Forward
// ============================================================

pub fn argmax<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<i64> {
    let shape = a.shape();
    assert!(dim < shape.len(), "argmax: dim out of range");

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

    let mut out_data = vec![0i64; out_size];

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                let v = a_data[idx].to_f32();
                if v > max_val {
                    max_val = v;
                    max_idx = d;
                }
            }
            let out_idx = o * inner + i;
            out_data[out_idx] = max_idx as i64;
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
// 2. ArgMin Forward
// ============================================================

pub fn argmin<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<i64> {
    let shape = a.shape();
    assert!(dim < shape.len(), "argmin: dim out of range");

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

    let mut out_data = vec![0i64; out_size];

    for o in 0..outer {
        for i in 0..inner {
            let mut min_val = f32::INFINITY;
            let mut min_idx = 0;
            let base = o * stride + i;
            for d in 0..dim_size {
                let idx = base + d * inner;
                let v = a_data[idx].to_f32();
                if v < min_val {
                    min_val = v;
                    min_idx = d;
                }
            }
            let out_idx = o * inner + i;
            out_data[out_idx] = min_idx as i64;
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
// 3. ArgMax Op (独立实现)
// ============================================================

pub struct ArgMaxOp;

impl ArgMaxOp {
    pub fn name(&self) -> &'static str {
        "argmax"
    }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Tensor<i64> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        argmax(inputs[0], dim, keepdim)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        _grad: &Tensor<i64>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i64>> {
        vec![]
    }
}

// ============================================================
// 4. ArgMin Op (独立实现)
// ============================================================

pub struct ArgMinOp;

impl ArgMinOp {
    pub fn name(&self) -> &'static str {
        "argmin"
    }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Tensor<i64> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        argmin(inputs[0], dim, keepdim)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        _grad: &Tensor<i64>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i64>> {
        vec![]
    }
}

// ============================================================
// 5. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_1d() {
        let a = Tensor::new(vec![1.0, 5.0, 3.0, 4.0], &[4]);
        let c = argmax(&a, 0, false);
        assert_eq!(c.data(), &[1]);
    }

    #[test]
    fn test_argmax_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = argmax(&a, 1, false);
        assert_eq!(c.data(), &[2, 2]);
    }

    #[test]
    fn test_argmax_keepdim() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = argmax(&a, 0, true);
        assert_eq!(c.shape(), &[1, 2]);
        assert_eq!(c.data(), &[1, 1]);
    }

    #[test]
    fn test_argmin_1d() {
        let a = Tensor::new(vec![1.0, 5.0, 3.0, 4.0], &[4]);
        let c = argmin(&a, 0, false);
        assert_eq!(c.data(), &[0]);
    }

    #[test]
    fn test_argmax_op() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let op = ArgMaxOp;
        let attrs = OpAttrs::new().with_int("dim", 0);
        let c = op.forward(&[&a], &attrs);
        assert_eq!(c.data(), &[3]);
    }

    #[test]
    fn test_argmax_f64() {
        let a = Tensor::new(vec![1.0, 5.0, 3.0, 4.0], &[4]);
        let c = argmax(&a, 0, false);
        assert_eq!(c.data(), &[1]);
    }
}
