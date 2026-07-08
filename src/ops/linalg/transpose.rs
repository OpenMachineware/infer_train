use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn transpose<T: DType + Send + Sync>(a: &Tensor<T>) -> Tensor<T> {
    let shape = a.shape();
    assert!(shape.len() >= 2, "transpose requires at least 2D tensor");

    let mut new_shape = shape.to_vec();
    let last = new_shape.len() - 1;
    new_shape.swap(last - 1, last);

    let rows = shape[last - 1];
    let cols = shape[last];
    let batch_stride = rows * cols;

    let data: Vec<T> = a
        .data()
        .par_chunks(batch_stride)
        .flat_map(|chunk| {
            let mut transposed = vec![T::from_f32(0.0); batch_stride];
            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = chunk[i * cols + j];
                }
            }
            transposed
        })
        .collect();

    Tensor::new(data, &new_shape)
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn transpose_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    // 转置的梯度就是再次转置
    vec![transpose(grad_output)]
}

// ============================================================
// 3. 量化 Forward
// ============================================================

pub fn quantized_transpose(a: &Tensor<i8>) -> Tensor<i8> {
    let shape = a.shape();
    assert!(
        shape.len() >= 2,
        "quantized_transpose requires at least 2D tensor"
    );

    let mut new_shape = shape.to_vec();
    let last = new_shape.len() - 1;
    new_shape.swap(last - 1, last);

    let rows = shape[last - 1];
    let cols = shape[last];
    let batch_stride = rows * cols;

    let data: Vec<i8> = a
        .data()
        .par_chunks(batch_stride)
        .flat_map(|chunk| {
            let mut transposed = vec![0i8; batch_stride];
            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = chunk[i * cols + j];
                }
            }
            transposed
        })
        .collect();

    let scale = a.scale().unwrap_or(1.0);
    let zero = a.zero_point().unwrap_or(0.0);
    Tensor::<i8>::new_quantized(data, &new_shape, scale, zero)
}

// ============================================================
// 4. 量化 Backward
// ============================================================

pub fn quantized_transpose_backward(
    grad_output: &Tensor<i8>,
) -> Vec<Tensor<i8>> {
    vec![quantized_transpose(grad_output)]
}

// ============================================================
// 5. Operator Trait 实现
// ============================================================

pub struct TransposeOp;

impl<T: DType + Send + Sync> Operator<T> for TransposeOp {
    fn name(&self) -> &'static str {
        "transpose"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        transpose(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        transpose_backward(grad)
    }
}

pub struct QuantizedTransposeOp;

impl Operator<i8> for QuantizedTransposeOp {
    fn name(&self) -> &'static str {
        "quantized_transpose"
    }
    fn forward(&self, inputs: &[&Tensor<i8>], _attrs: &OpAttrs) -> Tensor<i8> {
        assert_eq!(inputs.len(), 1);
        quantized_transpose(inputs[0])
    }
    fn backward(
        &self,
        grad: &Tensor<i8>,
        _inputs: &[&Tensor<i8>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<i8>> {
        quantized_transpose_backward(grad)
    }
    fn supports_quantized(&self) -> bool {
        true
    }
}

// ============================================================
// 6. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = transpose(&a);
        assert_eq!(c.data(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_transpose_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = transpose(&a);
        assert_eq!(c.data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_transpose_batch() {
        let a = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        );
        let c = transpose(&a);
        // batch 0: [[1,2],[3,4]] → [[1,3],[2,4]]
        // batch 1: [[5,6],[7,8]] → [[5,7],[6,8]]
        assert_eq!(c.data(), &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
        assert_eq!(c.shape(), &[2, 2, 2]);
    }
}
