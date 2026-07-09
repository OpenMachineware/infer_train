// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Eye (Identity Matrix)
// ============================================================

pub fn eye<T: DType + Send + Sync>(n: usize, m: Option<usize>) -> Tensor<T> {
    let m = m.unwrap_or(n);
    let mut data = vec![T::zero(); n * m];
    for i in 0..n.min(m) {
        data[i * m + i] = T::one();
    }
    Tensor::new(data, &[n, m])
}

// ============================================================
// Diag (Diagonal Matrix / Extract Diagonal)
// ============================================================

pub fn diag<T: DType + Send + Sync>(input: &Tensor<T>) -> Tensor<T> {
    let shape = input.shape();
    let n = shape[0];
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = input.data()[i];
    }
    Tensor::new(data, &[n, n])
}

// ============================================================
// Operators
// ============================================================

pub struct EyeOp;

impl<T: DType + Send + Sync> Operator<T> for EyeOp {
    fn name(&self) -> &'static str {
        "eye"
    }
    fn forward(&self, _inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        let n = attrs.get_int("n").unwrap_or(3) as usize;
        let m = attrs.get_int("m").map(|v| v as usize);
        eye::<T>(n, m)
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

pub struct DiagOp;

impl<T: DType + Send + Sync> Operator<T> for DiagOp {
    fn name(&self) -> &'static str {
        "diag"
    }
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 1);
        diag(inputs[0])
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
    fn test_eye() {
        let c = eye::<f32>(3, None);
        assert_eq!(c.shape(), &[3, 3]);
        assert_eq!(c.data(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_eye_rect() {
        let c = eye::<f32>(3, Some(2));
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.data(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_diag() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let c = diag(&input);
        assert_eq!(c.shape(), &[3, 3]);
        assert_eq!(c.data(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }
}
