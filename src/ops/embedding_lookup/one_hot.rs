// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
// use crate::ops::registry::{OpAttrs};

// ============================================================
// 1. One-Hot Forward
// ============================================================

pub fn one_hot<T: DType + Send + Sync>(
    indices: &Tensor<i64>,
    num_classes: usize,
) -> Tensor<T> {
    let indices_data = indices.data();
    let shape = indices.shape();

    let mut out_shape = shape.to_vec();
    out_shape.push(num_classes);
    let out_size: usize = out_shape.iter().product();
    let mut out_data = vec![T::from_f32(0.0); out_size];

    let last_dim_stride = 1;
    let batch_stride = num_classes;

    for (i, &idx) in indices_data.iter().enumerate() {
        let idx_usize = idx as usize;
        assert!(idx_usize < num_classes, "Index out of range");
        let out_idx = i * batch_stride + idx_usize * last_dim_stride;
        out_data[out_idx] = T::from_f32(1.0);
    }

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// 2. OneHot Op (独立实现)
// ============================================================

pub struct OneHotOp;

impl OneHotOp {
    pub fn name(&self) -> &'static str { "one_hot" }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        indices: &Tensor<i64>,
        num_classes: usize,
    ) -> Tensor<T> {
        one_hot(indices, num_classes)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        _grad: &Tensor<T>,
        _indices: &Tensor<i64>,
        _num_classes: usize,
    ) -> Vec<Tensor<T>> {
        vec![]
    }
}

// ============================================================
// 3. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot() {
        let indices = Tensor::new(vec![0, 2, 1, 0], &[4]);
        let c = one_hot::<f32>(&indices, 3);
        assert_eq!(c.shape(), &[4, 3]);
        assert_eq!(c.data(), &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_one_hot_2d() {
        let indices = Tensor::new(vec![0, 1, 2, 0], &[2, 2]);
        let c = one_hot::<f32>(&indices, 3);
        assert_eq!(c.shape(), &[2, 2, 3]);
    }
}
