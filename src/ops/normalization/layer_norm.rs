// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn layer_norm<T: DType + Send + Sync>(
    x: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    eps: f32,
) -> Tensor<T> {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let outer: usize = shape[..shape.len() - 1].iter().product();

    let x_data = x.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();

    let mut out_data = vec![T::from_f32(0.0); x.len()];

    for o in 0..outer {
        let base = o * last_dim;

        // 计算 mean
        let mut mean = 0.0;
        for i in 0..last_dim {
            mean += x_data[base + i].to_f32();
        }
        mean /= last_dim as f32;

        // 计算 variance
        let mut var = 0.0;
        for i in 0..last_dim {
            let diff = x_data[base + i].to_f32() - mean;
            var += diff * diff;
        }
        var /= last_dim as f32;

        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..last_dim {
            let idx = base + i;
            let norm = (x_data[idx].to_f32() - mean) * inv_std;
            out_data[idx] = T::from_f32(
                norm * gamma_data[i].to_f32() + beta_data[i].to_f32(),
            );
        }
    }

    Tensor::new(out_data, shape)
}

// ============================================================
// 2. 浮点泛型 Backward
// ============================================================

pub fn layer_norm_backward<T: DType + Send + Sync>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    eps: f32,
) -> Vec<Tensor<T>> {
    let shape = input.shape();
    let last_dim = shape[shape.len() - 1];
    let outer: usize = shape[..shape.len() - 1].iter().product();

    let grad_data = grad_output.data();
    let x_data = input.data();
    let gamma_data = weight.data();

    let mut grad_input = vec![T::zero(); input.len()];
    let mut grad_weight = vec![T::zero(); last_dim];
    let mut grad_bias = vec![T::zero(); last_dim];

    for o in 0..outer {
        let base = o * last_dim;

        let mut mean = 0.0;
        for i in 0..last_dim {
            mean += x_data[base + i].to_f32();
        }
        mean /= last_dim as f32;

        let mut var = 0.0;
        for i in 0..last_dim {
            let diff = x_data[base + i].to_f32() - mean;
            var += diff * diff;
        }
        var /= last_dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..last_dim {
            let idx = base + i;
            let x = x_data[idx].to_f32();
            let grad = grad_data[idx].to_f32();
            let gamma = gamma_data[i].to_f32();
            let norm = (x - mean) * inv_std;

            grad_input[idx] = T::from_f32(grad * gamma * inv_std);
            grad_weight[i] = T::from_f32(grad_weight[i].to_f32() + grad * norm);
            grad_bias[i] = T::from_f32(grad_bias[i].to_f32() + grad);
        }
    }

    vec![
        Tensor::new(grad_input, shape),
        Tensor::new(grad_weight, &[last_dim]),
        Tensor::new(grad_bias, &[last_dim]),
    ]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct LayerNormOp;

impl<T: DType + Send + Sync> Operator<T> for LayerNormOp {
    fn name(&self) -> &'static str {
        "layer_norm"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 3);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        layer_norm(inputs[0], inputs[1], inputs[2], eps)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 3);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        layer_norm_backward(grad, inputs[0], inputs[1], eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let gamma = Tensor::new(vec![1.0, 1.0, 1.0], &[3]);
        let beta = Tensor::new(vec![0.0, 0.0, 0.0], &[3]);
        let c = layer_norm(&x, &gamma, &beta, 1e-5);
        assert_eq!(c.shape(), &[2, 3]);
        // 每行均值为0，方差为1
        let row0_mean: f32 =
            c.data()[0..3].iter().map(|&x| x.to_f32()).sum::<f32>() / 3.0;
        assert!(f32::abs(row0_mean) < 0.001);
    }
}
