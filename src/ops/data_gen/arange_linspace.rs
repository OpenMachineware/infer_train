// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. Arange (等差序列)
// ============================================================

pub fn arange<T: DType + Send + Sync>(start: f32, end: f32, step: f32) -> Tensor<T> {
    assert!(step > 0.0, "arange: step must be positive");
    let len = ((end - start) / step).ceil() as usize;
    let data: Vec<T> = (0..len)
        .map(|i| T::from_f32(start + i as f32 * step))
        .collect();
    Tensor::new(data, &[len])
}

// ============================================================
// 2. Linspace (线性间隔)
// ============================================================

pub fn linspace<T: DType + Send + Sync>(start: f32, end: f32, steps: usize) -> Tensor<T> {
    assert!(steps > 1, "linspace: steps must be > 1");
    let step = (end - start) / (steps - 1) as f32;
    let data: Vec<T> = (0..steps)
        .map(|i| T::from_f32(start + i as f32 * step))
        .collect();
    Tensor::new(data, &[steps])
}

// ============================================================
// 3. Operators
// ============================================================

pub struct ArangeOp;

impl<T: DType + Send + Sync> Operator<T> for ArangeOp {
    fn name(&self) -> &'static str { "arange" }
    fn forward(&self, _inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        let start = attrs.get_float("start").unwrap_or(0.0);
        let end = attrs.get_float("end").unwrap_or(1.0);
        let step = attrs.get_float("step").unwrap_or(1.0);
        arange::<T>(start, end, step)
    }
    fn backward(&self, _grad: &Tensor<T>, _inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        vec![]
    }
}

pub struct LinspaceOp;

impl<T: DType + Send + Sync> Operator<T> for LinspaceOp {
    fn name(&self) -> &'static str { "linspace" }
    fn forward(&self, _inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        let start = attrs.get_float("start").unwrap_or(0.0);
        let end = attrs.get_float("end").unwrap_or(1.0);
        let steps = attrs.get_int("steps").unwrap_or(10) as usize;
        linspace::<T>(start, end, steps)
    }
    fn backward(&self, _grad: &Tensor<T>, _inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arange() {
        let c = arange::<f32>(0.0, 5.0, 1.0);
        assert_eq!(c.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.shape(), &[5]);
    }

    #[test]
    fn test_arange_f16() {
        use half::f16;
        let c = arange::<f16>(0.0, 3.0, 0.5);
        assert_eq!(c.shape(), &[6]);
        assert_eq!(c.data()[0], f16::from_f32(0.0));
        assert_eq!(c.data()[5], f16::from_f32(2.5));
    }

    #[test]
    fn test_linspace() {
        let c = linspace::<f32>(0.0, 10.0, 5);
        assert_eq!(c.data(), &[0.0, 2.5, 5.0, 7.5, 10.0]);
        assert_eq!(c.shape(), &[5]);
    }
}
