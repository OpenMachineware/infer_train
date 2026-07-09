use rayon::prelude::*;
// use rand::Rng;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rand::distributions::{Distribution, Uniform};
use rand_distr::StandardNormal;

// ============================================================
// Rand (均匀分布 [0, 1))
// ============================================================

pub fn rand<T: DType + Send + Sync>(shape: &[usize]) -> Tensor<T> {
    let size: usize = shape.iter().product();
    let uniform = Uniform::new(0.0_f32, 1.0_f32);

    // 每个线程独立获取 rng
    let data: Vec<T> = (0..size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            T::from_f32(uniform.sample(&mut rng))
        })
        .collect();

    Tensor::new(data, shape)
}

// ============================================================
// Randn (正态分布 N(0, 1))
// ============================================================

pub fn randn<T: DType + Send + Sync>(shape: &[usize]) -> Tensor<T> {
    let size: usize = shape.iter().product();

    // 每个线程独立获取 rng
    let data: Vec<T> = (0..size)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            T::from_f32(StandardNormal.sample(&mut rng))
        })
        .collect();

    Tensor::new(data, shape)
}

// ============================================================
// Operators
// ============================================================

pub struct RandOp;

impl<T: DType + Send + Sync> Operator<T> for RandOp {
    fn name(&self) -> &'static str {
        "rand"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        let shape = inputs[0].shape();
        rand::<T>(shape)
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }
}

pub struct RandnOp;

impl<T: DType + Send + Sync> Operator<T> for RandnOp {
    fn name(&self) -> &'static str {
        "randn"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        let shape = inputs[0].shape();
        randn::<T>(shape)
    }
    fn backward(
        &self,
        _grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let c = rand::<f32>(&[10]);
        assert_eq!(c.shape(), &[10]);
        for &v in c.data() {
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_randn() {
        let c = randn::<f32>(&[10]);
        assert_eq!(c.shape(), &[10]);
    }
}
