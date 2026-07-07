use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn permute<T: DType + Send + Sync>(a: &Tensor<T>, dims: &[usize]) -> Tensor<T> {
    let shape = a.shape();
    assert_eq!(shape.len(), dims.len(), "permute: dims length must match rank");

    // 验证 dims 有效
    let mut seen = vec![false; shape.len()];
    for &d in dims {
        assert!(d < shape.len(), "permute: dim {} out of range", d);
        assert!(!seen[d], "permute: duplicate dim {}", d);
        seen[d] = true;
    }

    let new_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();

    // 计算原始 shape 的步长
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }

    // 计算新 shape 的步长 (按照 dims 顺序)
    let mut new_stride = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        new_stride[i] = new_stride[i + 1] * new_shape[i + 1];
    }

    let total: usize = a.data().len();
    let a_data = a.data();

    // 遍历新张量的每个位置，根据新步长解出维度索引，再映射回原始位置
    let data: Vec<T> = (0..total).into_par_iter()
        .map(|new_idx| {
            let mut old_idx = 0;
            let mut rem = new_idx;
            // 按照新维度的顺序
            for i in 0..dims.len() {
                let dim_idx = rem / new_stride[i];
                rem %= new_stride[i];
                // 原始位置 = 维度索引 * 原始步长[原始维度]
                old_idx += dim_idx * stride[dims[i]];
            }
            a_data[old_idx]
        })
        .collect();

    Tensor::new(data, &new_shape)
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn permute_backward<T: DType>(
    grad_output: &Tensor<T>,
    dims: &[usize],
) -> Vec<Tensor<T>> {
    let mut inv_dims = vec![0; dims.len()];
    for (i, &d) in dims.iter().enumerate() {
        inv_dims[d] = i;
    }
    vec![permute(grad_output, &inv_dims)]
}

// ============================================================
// 3. 另一种实现：用循环构建索引（更易懂）
// ============================================================

pub fn permute_loop<T: DType + Clone>(a: &Tensor<T>, dims: &[usize]) -> Tensor<T> {
    let shape = a.shape();
    let new_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();

    // 计算原始步长
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }

    let total: usize = new_shape.iter().product();
    let mut data = vec![T::from_f32(0.0); total];
    let a_data = a.data();

    // 计算新步长
    let mut new_stride = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        new_stride[i] = new_stride[i + 1] * new_shape[i + 1];
    }

    for new_idx in 0..total {
        let mut old_idx = 0;
        let mut rem = new_idx;
        for i in 0..dims.len() {
            let dim_idx = rem / new_stride[i];
            rem %= new_stride[i];
            old_idx += dim_idx * stride[dims[i]];
        }
        data[new_idx] = a_data[old_idx];
    }

    Tensor::new(data, &new_shape)
}

// ============================================================
// 4. Operator Trait 实现
// ============================================================

pub struct PermuteOp;

impl<T: DType + Send + Sync> Operator<T> for PermuteOp {
    fn name(&self) -> &'static str { "permute" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dims = attrs.get_int_list("dims")
            .expect("permute requires dims attribute")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        permute(inputs[0], &dims)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 1);
        let dims = attrs.get_int_list("dims")
            .expect("permute requires dims attribute")
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        permute_backward(grad, &dims)
    }
}

// ============================================================
// 5. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // 原始: [[1,2],[3,4]]
        // dims [1,0]: 交换维度 → [[1,3],[2,4]]
        // 数据: [1,3,2,4]
        let c = permute(&a, &[1, 0]);
        assert_eq!(c.data(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_permute_2d_loop() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = permute_loop(&a, &[1, 0]);
        assert_eq!(c.data(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_permute_3d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        // dims [2,0,1]: (d2, d0, d1)
        // 期望: [1,3,5,7,2,4,6,8]
        let c = permute(&a, &[2, 0, 1]);
        assert_eq!(c.data(), &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(c.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_permute_3x2_to_2x3() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // 原始: [[1,2],[3,4],[5,6]]
        // dims [1,0]: → [[1,3,5],[2,4,6]]
        // 数据: [1,3,5,2,4,6]
        let c = permute(&a, &[1, 0]);
        assert_eq!(c.data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        assert_eq!(c.shape(), &[2, 3]);
    }
}
