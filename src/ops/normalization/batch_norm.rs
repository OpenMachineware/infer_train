// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. 浮点泛型 Forward (推理模式)
// ============================================================

pub fn batch_norm<T: DType + Send + Sync>(
    x: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    eps: f32,
) -> Tensor<T> {
    let shape = x.shape();
    assert!(shape.len() >= 2, "batch_norm requires at least 2D tensor");

    let c = shape[1]; // channels
    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];

    let x_data = x.data();
    let mean_data = running_mean.data();
    let var_data = running_var.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();

    let mut out_data = vec![T::from_f32(0.0); x.len()];

    for b in 0..batch {
        for ch in 0..c {
            let mean = mean_data[ch].to_f32();
            let var = var_data[ch].to_f32();
            let inv_std = 1.0 / (var + eps).sqrt();
            let g = gamma_data[ch].to_f32();
            let bet = beta_data[ch].to_f32();

            let base = (b * c + ch) * spatial;
            for s in 0..spatial {
                let idx = base + s;
                let norm = (x_data[idx].to_f32() - mean) * inv_std;
                out_data[idx] = T::from_f32(g * norm + bet);
            }
        }
    }

    Tensor::new(out_data, shape)
}

// ============================================================
// 2. 浮点泛型 Backward (训练模式)
// ============================================================

pub fn batch_norm_backward<T: DType + Send + Sync>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    running_mean: &Tensor<T>,
    running_var: &Tensor<T>,
    eps: f32,
) -> Vec<Tensor<T>> {
    let shape = input.shape();
    let c = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];

    let grad_data = grad_output.data();
    let x_data = input.data();
    let mean_data = running_mean.data();
    let var_data = running_var.data();
    let gamma_data = weight.data();

    let mut grad_input = vec![T::zero(); input.len()];
    let mut grad_weight = vec![T::zero(); c];
    let mut grad_bias = vec![T::zero(); c];

    for b in 0..batch {
        for ch in 0..c {
            let mean = mean_data[ch].to_f32();
            let var = var_data[ch].to_f32();
            let inv_std = 1.0 / (var + eps).sqrt();
            let gamma = gamma_data[ch].to_f32();
            let base = (b * c + ch) * spatial;

            for s in 0..spatial {
                let idx = base + s;
                let x = x_data[idx].to_f32();
                let grad = grad_data[idx].to_f32();

                grad_input[idx] = T::from_f32(grad * gamma * inv_std);
                grad_weight[ch] = T::from_f32(grad_weight[ch].to_f32() + grad * (x - mean) * inv_std);
                grad_bias[ch] = T::from_f32(grad_bias[ch].to_f32() + grad);
            }
        }
    }

    vec![
        Tensor::new(grad_input, shape),
        Tensor::new(grad_weight, &[c]),
        Tensor::new(grad_bias, &[c]),
    ]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct BatchNormOp;

impl<T: DType + Send + Sync> Operator<T> for BatchNormOp {
    fn name(&self) -> &'static str { "batch_norm" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 5);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        batch_norm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 5);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        batch_norm_backward(grad, inputs[0], inputs[1], inputs[3], inputs[4], eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_2d() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let gamma = Tensor::new(vec![1.0, 1.0, 1.0], &[3]);
        let beta = Tensor::new(vec![0.0, 0.0, 0.0], &[3]);
        let mean = Tensor::new(vec![0.0, 0.0, 0.0], &[3]);
        let var = Tensor::new(vec![1.0, 1.0, 1.0], &[3]);
        let c = batch_norm(&x, &gamma, &beta, &mean, &var, 1e-5);
        assert_eq!(c.shape(), &[2, 3]);
    }
}
