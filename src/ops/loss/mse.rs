use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;
use rayon::prelude::*;

// ============================================================
// 浮点泛型 Forward
// ============================================================

pub fn mse<T: DType + Send + Sync>(
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: bool,
) -> Tensor<T> {
    assert_eq!(pred.shape(), target.shape(), "MSE: shape mismatch");

    let pred_data = pred.data();
    let target_data = target.data();

    let sum_sq: f32 = pred_data
        .par_iter()
        .zip(target_data.par_iter())
        .map(|(&p, &t)| {
            let diff = p.to_f32() - t.to_f32();
            diff * diff
        })
        .sum();

    let result = if reduction && pred.len() > 0 {
        sum_sq / pred.len() as f32
    } else {
        sum_sq
    };

    Tensor::new(vec![T::from_f32(result)], &[1])
}

// ============================================================
// 浮点泛型 Backward
// ============================================================

pub fn mse_backward<T: DType>(
    grad_output: &Tensor<T>,
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: bool,
) -> Vec<Tensor<T>> {
    let grad_val = grad_output.data()[0].to_f32();
    let scale = if reduction && pred.len() > 0 {
        2.0 * grad_val / pred.len() as f32
    } else {
        2.0 * grad_val
    };

    let grad_pred: Vec<T> = pred
        .data()
        .par_iter()
        .zip(target.data().par_iter())
        .map(|(&p, &t)| T::from_f32((p.to_f32() - t.to_f32()) * scale))
        .collect();

    vec![Tensor::new(grad_pred, pred.shape())]
}

// ============================================================
// Operator Trait 实现
// ============================================================

pub struct MseOp;

impl<T: DType + Send + Sync> Operator<T> for MseOp {
    fn name(&self) -> &'static str {
        "mse"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        let reduction = attrs.get_bool("reduction").unwrap_or(true);
        mse(inputs[0], inputs[1], reduction)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        let reduction = attrs.get_bool("reduction").unwrap_or(true);
        mse_backward(grad, inputs[0], inputs[1], reduction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let target = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let loss = mse(&pred, &target, true);
        let eps = 0.001;
        assert!(f32::abs(loss.data()[0].to_f32() - 0.0) < eps);
    }

    #[test]
    fn test_mse_diff() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], &[3]);
        let target = Tensor::new(vec![0.0, 0.0, 0.0], &[3]);
        let loss = mse(&pred, &target, true);
        // (1+4+9)/3 = 14/3 = 4.666...
        assert!((loss.data()[0].to_f32() - 4.666).abs() < 0.01);
    }
}
