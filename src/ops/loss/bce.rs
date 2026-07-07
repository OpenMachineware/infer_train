use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn bce<T: DType + Send + Sync>(
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: bool,
    eps: f32,
) -> Tensor<T> {
    assert_eq!(pred.shape(), target.shape(), "BCE: shape mismatch");

    let pred_data = pred.data();
    let target_data = target.data();

    let sum_loss: f32 = pred_data.par_iter()
        .zip(target_data.par_iter())
        .map(|(&p, &t)| {
            let p_clamped = p.to_f32().clamp(eps, 1.0 - eps);
            let t_val = t.to_f32();
            - (t_val * p_clamped.ln() + (1.0 - t_val) * (1.0 - p_clamped).ln())
        })
        .sum();

    let result = if reduction && pred.len() > 0 {
        sum_loss / pred.len() as f32
    } else {
        sum_loss
    };

    Tensor::new(vec![T::from_f32(result)], &[1])
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn bce_backward<T: DType>(
    grad_output: &Tensor<T>,
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: bool,
    eps: f32,
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
            let p_clamped = p.to_f32().clamp(eps, 1.0 - eps);
            let t_val = t.to_f32();
            let grad = (p_clamped - t_val) / (p_clamped * (1.0 - p_clamped));
            T::from_f32(scale * grad)
        })
        .collect();

    vec![Tensor::new(grad_pred, pred.shape())]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct BceOp;

impl<T: DType + Send + Sync> Operator<T> for BceOp {
    fn name(&self) -> &'static str { "bce" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        let reduction = attrs.get_bool("reduction").unwrap_or(true);
        let eps = attrs.get_float("eps").unwrap_or(1e-7);
        bce(inputs[0], inputs[1], reduction, eps)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        let reduction = attrs.get_bool("reduction").unwrap_or(true);
        let eps = attrs.get_float("eps").unwrap_or(1e-7);
        bce_backward(grad, inputs[0], inputs[1], reduction, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bce() {
        let pred = Tensor::new(vec![0.9, 0.1, 0.8], &[3]);
        let target = Tensor::new(vec![1.0, 0.0, 1.0], &[3]);
        let loss = bce(&pred, &target, true, 1e-7);
        assert_eq!(loss.shape(), &[1]);
        assert!(loss.data()[0].to_f32() > 0.0);
    }
}
