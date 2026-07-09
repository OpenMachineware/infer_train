// use rayon::prelude::*;
use super::sum::{sum, sum_backward};
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn mean<T: DType + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Tensor<T> {
    let shape = a.shape();
    let dim_size = shape[dim];
    let c = sum(a, dim, keepdim);

    let data: Vec<T> = c
        .data()
        .iter()
        .map(|&x| T::from_f32(x.to_f32() / dim_size as f32))
        .collect();

    Tensor::new(data, c.shape())
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn mean_backward<T: DType>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    dim: usize,
    keepdim: bool,
) -> Vec<Tensor<T>> {
    let shape = a.shape();
    let dim_size = shape[dim];
    let grads = sum_backward(grad_output, a, dim, keepdim);
    let grad = grads[0].clone();

    let data: Vec<T> = grad
        .data()
        .iter()
        .map(|&x| T::from_f32(x.to_f32() / dim_size as f32))
        .collect();

    vec![Tensor::new(data, grad.shape())]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct MeanOp;

impl<T: DType + Send + Sync> Operator<T> for MeanOp {
    fn name(&self) -> &'static str {
        "mean"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        let dim = attrs.get_int("dim").unwrap_or(0) as usize;
        let keepdim = attrs.get_bool("keepdim").unwrap_or(false);
        mean(inputs[0], dim, keepdim)
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
        mean_backward(grad, inputs[0], dim, keepdim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = mean(&a, 1, false);
        assert_eq!(c.data(), &[2.0, 5.0]);
        assert_eq!(c.shape(), &[2]);
    }
}
