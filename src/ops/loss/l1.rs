use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn l1<T: DType + Send + Sync>(
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: bool,
) -> Tensor<T> {
    assert_eq!(pred.shape(), target.shape(), "L1: shape mismatch");

    let pred_data = pred.data();
    let target_data = target.data();

    let sum_abs: f32 = pred_data.par_iter()
        .zip(target_data.par_iter())
        .map(|(&p, &t)| (p.to_f32() - t.to_f32()).abs())
        .sum();

    let result = if reduction && pred.len() > 0 {
        sum_abs / pred.len() as f32
    } else {
        sum_abs
    };

    Tensor::new(vec![T::from_f32(result)], &[1])
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn l1_backward<T: DType>(
    grad_output: &Tensor<T>,
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: bool,
) -> Vec<Tensor<T>> {
    let grad_val = grad_output.data()[0].to_f32();
    let scale = if reduction && pred.len() > 0 {
        grad_val / pred.len() as f32
    } else {
        grad_val
    };

    let grad_pred: Vec<T> = pred.data()
        .par_iter()
        .zip(target.data().par_iter())
        .map(|(&p, &t)| {
            let diff = p.to_f32() - t.to_f32();
            T::from_f32(scale * diff.signum())
        })
        .collect();

    vec![Tensor::new(grad_pred, pred.shape())]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct L1Op;

impl<T: DType + Send + Sync> Operator<T> for L1Op {
    fn name(&self) -> &'static str { "l1" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        let reduction = attrs.get_bool("reduction").unwrap_or(true);
        l1(inputs[0], inputs[1], reduction)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        let reduction = attrs.get_bool("reduction").unwrap_or(true);
        l1_backward(grad, inputs[0], inputs[1], reduction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let target = Tensor::new(vec![0.0, 0.0, 0.0], &[3]);
        let loss = l1(&pred, &target, true);
        // (1+2+3)/3 = 2
        assert!((loss.data()[0].to_f32() - 2.0).abs() < 0.01);
    }
}
