// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
// use crate::ops::registry::{OpAttrs};

// ============================================================
// Float Generic Forward
// ============================================================

pub fn cross_entropy<T: DType + Send + Sync>(
    logits: &Tensor<T>,
    targets: &Tensor<i64>,
    reduction: bool,
) -> Tensor<T> {
    let shape = logits.shape();
    assert!(shape.len() >= 2, "cross_entropy requires at least 2D logits");

    let batch = shape[0];
    let num_classes = shape[1];
    let spatial: usize = shape[2..].iter().product();

    let logits_data = logits.data();
    let targets_data = targets.data();

    let mut loss_sum = T::from_f32(0.0);
    let mut count = 0;

    for b in 0..batch {
        for s in 0..spatial {
            let target_idx = targets_data[b * spatial + s] as usize;
            assert!(target_idx < num_classes, "Target out of range");

            let base = (b * num_classes + 0) * spatial + s;

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let idx = base + c * spatial;
                let v = logits_data[idx].to_f32();
                if v > max_val {
                    max_val = v;
                }
            }

            let mut sum_exp = 0.0;
            for c in 0..num_classes {
                let idx = base + c * spatial;
                sum_exp += (logits_data[idx].to_f32() - max_val).exp();
            }
            let log_sum_exp = max_val + sum_exp.ln();

            let target_logit_idx = base + target_idx * spatial;
            let loss = log_sum_exp - logits_data[target_logit_idx].to_f32();
            loss_sum = loss_sum + T::from_f32(loss);
            count += 1;
        }
    }

    let result = if reduction && count > 0 {
        T::from_f32(loss_sum.to_f32() / count as f32)
    } else {
        loss_sum
    };

    Tensor::new(vec![result], &[1])
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn cross_entropy_backward<T: DType>(
    grad_output: &Tensor<T>,
    logits: &Tensor<T>,
    targets: &Tensor<i64>,
    reduction: bool,
) -> Vec<Tensor<T>> {
    let shape = logits.shape();
    let batch = shape[0];
    let num_classes = shape[1];
    let spatial: usize = shape[2..].iter().product();

    let logits_data = logits.data();
    let targets_data = targets.data();
    let grad = grad_output.data()[0].to_f32();

    let mut grad_logits = vec![T::from_f32(0.0); logits.len()];

    let scale = if reduction && batch > 0 {
        grad / (batch * spatial) as f32
    } else {
        grad
    };

    for b in 0..batch {
        for s in 0..spatial {
            let target_idx = targets_data[b * spatial + s] as usize;
            let base = (b * num_classes + 0) * spatial + s;

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let idx = base + c * spatial;
                let v = logits_data[idx].to_f32();
                if v > max_val {
                    max_val = v;
                }
            }

            let mut sum_exp = 0.0;
            for c in 0..num_classes {
                let idx = base + c * spatial;
                sum_exp += (logits_data[idx].to_f32() - max_val).exp();
            }

            for c in 0..num_classes {
                let idx = base + c * spatial;
                let softmax =
                    (logits_data[idx].to_f32() - max_val).exp() / sum_exp;
                let grad_val = if c == target_idx {
                    scale * (softmax - 1.0)
                } else {
                    scale * softmax
                };
                grad_logits[idx] = T::from_f32(grad_val);
            }
        }
    }

    vec![Tensor::new(grad_logits, shape)]
}

// ============================================================
// CrossEntropy Op (Standalone Implementation)
// ============================================================

pub struct CrossEntropyOp;

impl CrossEntropyOp {
    pub fn name(&self) -> &'static str {
        "cross_entropy"
    }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        logits: &Tensor<T>,
        targets: &Tensor<i64>,
        reduction: bool,
    ) -> Tensor<T> {
        cross_entropy(logits, targets, reduction)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        grad: &Tensor<T>,
        logits: &Tensor<T>,
        targets: &Tensor<i64>,
        reduction: bool,
    ) -> Vec<Tensor<T>> {
        cross_entropy_backward(grad, logits, targets, reduction)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy() {
        let logits = Tensor::new(vec![2.0, 1.0, 0.1, 0.5, 2.0, 0.3], &[2, 3]);
        let targets = Tensor::new(vec![0, 1], &[2]);
        let loss = cross_entropy(&logits, &targets, true);
        assert_eq!(loss.shape(), &[1]);
        assert!(loss.data()[0].to_f32() > 0.0);
    }

    #[test]
    fn test_cross_entropy_op() {
        let logits = Tensor::new(vec![2.0, 1.0, 0.1, 0.5, 2.0, 0.3], &[2, 3]);
        let targets = Tensor::new(vec![0, 1], &[2]);
        let op = CrossEntropyOp;
        let loss = op.forward(&logits, &targets, true);
        assert_eq!(loss.shape(), &[1]);
        assert!(loss.data()[0].to_f32() > 0.0);
    }
}
