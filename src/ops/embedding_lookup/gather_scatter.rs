// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
// use crate::ops::registry::{OpAttrs};

// ============================================================
// 1. Gather Forward
// ============================================================

pub fn gather<T: DType + Send + Sync>(
    input: &Tensor<T>,
    indices: &Tensor<i64>,
    dim: usize,
) -> Tensor<T> {
    let input_shape = input.shape();
    let indices_shape = indices.shape();
    assert!(dim < input_shape.len(), "gather: dim out of range");

    let indices_total: usize = indices_shape.iter().product();
    let mut out_shape = input_shape.to_vec();
    out_shape[dim] = indices_total;

    let outer: usize = input_shape[..dim].iter().product();
    let inner: usize = input_shape[dim + 1..].iter().product();
    let dim_size = input_shape[dim];

    let input_data = input.data();
    let indices_data = indices.data();
    let total = outer * indices_total * inner;
    let mut out_data = vec![T::from_f32(0.0); total];

    let mut out_pos = 0;
    for o in 0..outer {
        for i_idx in 0..indices_total {
            let idx_val = indices_data[i_idx] as usize;
            assert!(idx_val < dim_size, "Index out of range");
            let input_base = o * dim_size * inner + idx_val * inner;
            for inn in 0..inner {
                out_data[out_pos] = input_data[input_base + inn];
                out_pos += 1;
            }
        }
    }

    Tensor::new(out_data, &out_shape)
}

// ============================================================
// 2. Gather Backward
// ============================================================

pub fn gather_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// 3. Scatter Forward
// ============================================================

pub fn scatter<T: DType + Send + Sync>(
    input: &Tensor<T>,
    indices: &Tensor<i64>,
    src: &Tensor<T>,
    dim: usize,
) -> Tensor<T> {
    let input_shape = input.shape();
    let indices_shape = indices.shape();
    assert_eq!(
        indices_shape,
        src.shape(),
        "scatter: indices and src shape mismatch"
    );
    assert!(dim < input_shape.len(), "scatter: dim out of range");

    let mut out_data = input.data().to_vec();
    let indices_data = indices.data();
    let src_data = src.data();

    let outer: usize = input_shape[..dim].iter().product();
    let inner: usize = input_shape[dim + 1..].iter().product();
    let dim_size = input_shape[dim];
    let indices_total: usize = indices_shape.iter().product();

    let mut src_pos = 0;
    for o in 0..outer {
        for i_idx in 0..indices_total {
            let idx_val = indices_data[i_idx] as usize;
            assert!(idx_val < dim_size, "Index out of range");
            let input_base = o * dim_size * inner + idx_val * inner;
            for inn in 0..inner {
                out_data[input_base + inn] = src_data[src_pos];
                src_pos += 1;
            }
        }
    }

    Tensor::new(out_data, input_shape)
}

// ============================================================
// 4. Scatter Backward
// ============================================================

pub fn scatter_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// 5. Gather Op
// ============================================================

pub struct GatherOp;

impl GatherOp {
    pub fn name(&self) -> &'static str {
        "gather"
    }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        input: &Tensor<T>,
        indices: &Tensor<i64>,
        dim: usize,
    ) -> Tensor<T> {
        gather(input, indices, dim)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        grad: &Tensor<T>,
        _input: &Tensor<T>,
        _indices: &Tensor<i64>,
        _dim: usize,
    ) -> Vec<Tensor<T>> {
        gather_backward(grad)
    }
}

// ============================================================
// 6. Scatter Op
// ============================================================

pub struct ScatterOp;

impl ScatterOp {
    pub fn name(&self) -> &'static str {
        "scatter"
    }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        input: &Tensor<T>,
        indices: &Tensor<i64>,
        src: &Tensor<T>,
        dim: usize,
    ) -> Tensor<T> {
        scatter(input, indices, src, dim)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        grad: &Tensor<T>,
        _input: &Tensor<T>,
        _indices: &Tensor<i64>,
        _src: &Tensor<T>,
        _dim: usize,
    ) -> Vec<Tensor<T>> {
        scatter_backward(grad)
    }
}

// ============================================================
// 7. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_1d() {
        let input = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], &[4]);
        let indices = Tensor::new(vec![0, 2, 3], &[3]);
        let c = gather(&input, &indices, 0);
        assert_eq!(c.data(), &[10.0, 30.0, 40.0]);
        assert_eq!(c.shape(), &[3]);
    }

    #[test]
    fn test_gather_2d() {
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        );
        let indices = Tensor::new(vec![0, 2], &[2]);
        let c = gather(&input, &indices, 0);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_gather_2d_dim1() {
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        );
        let indices = Tensor::new(vec![0, 2], &[2]);
        let c = gather(&input, &indices, 1);
        assert_eq!(c.data(), &[1.0, 3.0, 4.0, 6.0, 7.0, 9.0]);
        assert_eq!(c.shape(), &[3, 2]);
    }

    #[test]
    fn test_scatter() {
        let input = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], &[4]);
        let indices = Tensor::new(vec![0, 2, 3], &[3]);
        let src = Tensor::new(vec![10.0, 20.0, 30.0], &[3]);
        let c = scatter(&input, &indices, &src, 0);
        assert_eq!(c.data(), &[10.0, 0.0, 20.0, 30.0]);
    }

    #[test]
    fn test_gather_op() {
        let input = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], &[4]);
        let indices = Tensor::new(vec![0, 2, 3], &[3]);
        let op = GatherOp;
        let c = op.forward(&input, &indices, 0);
        assert_eq!(c.data(), &[10.0, 30.0, 40.0]);
    }
}
