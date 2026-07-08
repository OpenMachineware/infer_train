use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 1. Cast (类型转换)
// ============================================================

pub fn cast<T: DType + Send + Sync, U: DType + Send + Sync>(
    input: &Tensor<T>,
) -> Tensor<U> {
    let data: Vec<U> =
        input.data().par_iter().map(|&x| U::from_f32(x.to_f32())).collect();

    Tensor::new(data, input.shape())
}

// ============================================================
// 2. Cast Backward (反向传播是恒等映射)
// ============================================================

pub fn cast_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct CastOp;

impl<T: DType + Send + Sync> Operator<T> for CastOp {
    fn name(&self) -> &'static str {
        "cast"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        // 返回原值（实际 cast 需要指定目标类型）
        inputs[0].clone()
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        cast_backward(grad)
    }
}

// ============================================================
// 4. 便捷函数: 特定类型转换
// ============================================================

pub fn to_f32<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<f32> {
    cast::<T, f32>(input)
}

pub fn to_f64<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<f64> {
    cast::<T, f64>(input)
}

pub fn to_f16<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<half::f16> {
    use half::f16;
    cast::<T, f16>(input)
}

pub fn to_bf16<T: DType + Send + Sync>(
    input: &Tensor<T>,
) -> Tensor<half::bf16> {
    use half::bf16;
    cast::<T, bf16>(input)
}

pub fn to_i8<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<i8> {
    cast::<T, i8>(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_f32_to_f64() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = cast::<f32, f64>(&input);
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_f32() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = to_f32(&input);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cast_f64_to_f32() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = cast::<f64, f32>(&input);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0]);
    }
}
