use crate::dtype::DType;
use crate::tensor::Tensor;
use rayon::prelude::*;
// use crate::ops::registry::{OpAttrs};

// ============================================================
// 1. Select Forward (条件选择)
// ============================================================

pub fn select<T: DType + Send + Sync>(
    condition: &Tensor<i8>,
    true_val: &Tensor<T>,
    false_val: &Tensor<T>,
) -> Tensor<T> {
    assert_eq!(
        true_val.shape(),
        false_val.shape(),
        "select: shape mismatch between true and false"
    );
    assert_eq!(
        condition.len(),
        true_val.len(),
        "select: condition length must match tensor size"
    );

    let cond_data = condition.data();
    let true_data = true_val.data();
    let false_data = false_val.data();

    let data: Vec<T> = (0..cond_data.len())
        .into_par_iter()
        .map(|i| if cond_data[i] != 0 { true_data[i] } else { false_data[i] })
        .collect();

    Tensor::new(data, true_val.shape())
}

// ============================================================
// 2. Select Backward
// ============================================================

pub fn select_backward<T: DType>(
    grad_output: &Tensor<T>,
    condition: &Tensor<i8>,
) -> Vec<Tensor<T>> {
    let cond_data = condition.data();
    let grad_data = grad_output.data();

    let mut grad_true = vec![T::from_f32(0.0); grad_data.len()];
    let mut grad_false = vec![T::from_f32(0.0); grad_data.len()];

    for i in 0..grad_data.len() {
        if cond_data[i] != 0 {
            grad_true[i] = grad_data[i];
        } else {
            grad_false[i] = grad_data[i];
        }
    }

    vec![
        Tensor::new(grad_true, grad_output.shape()),
        Tensor::new(grad_false, grad_output.shape()),
    ]
}

// ============================================================
// 3. Select Op (独立实现)
// ============================================================

pub struct SelectOp;

impl SelectOp {
    pub fn name(&self) -> &'static str {
        "select"
    }
    pub fn forward<T: DType + Send + Sync>(
        &self,
        condition: &Tensor<i8>,
        true_val: &Tensor<T>,
        false_val: &Tensor<T>,
    ) -> Tensor<T> {
        select(condition, true_val, false_val)
    }
    pub fn backward<T: DType + Send + Sync>(
        &self,
        grad: &Tensor<T>,
        condition: &Tensor<i8>,
    ) -> Vec<Tensor<T>> {
        select_backward(grad, condition)
    }
}

// ============================================================
// 4. 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select() {
        let cond = Tensor::new(vec![1, 0, 1, 0], &[4]);
        let true_val = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let false_val = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[4]);
        let c = select(&cond, &true_val, &false_val);
        assert_eq!(c.data(), &[1.0, 6.0, 3.0, 8.0]);
    }

    #[test]
    fn test_select_op() {
        let cond = Tensor::new(vec![1, 0, 1, 0], &[4]);
        let true_val = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let false_val = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[4]);
        let op = SelectOp;
        let c = op.forward(&cond, &true_val, &false_val);
        assert_eq!(c.data(), &[1.0, 6.0, 3.0, 8.0]);
    }
}
