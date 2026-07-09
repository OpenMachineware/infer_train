// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn instance_norm<T: DType + Send + Sync>(
    x: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    eps: f32,
) -> Tensor<T> {
    let shape = x.shape();
    assert!(shape.len() >= 3, "instance_norm requires at least 3D tensor");

    let c = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];

    let x_data = x.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();

    let mut out_data = vec![T::from_f32(0.0); x.len()];

    for b in 0..batch {
        for ch in 0..c {
            let base = (b * c + ch) * spatial;

            // Compute mean
            let mut mean = 0.0;
            for s in 0..spatial {
                mean += x_data[base + s].to_f32();
            }
            mean /= spatial as f32;

            // Compute variance
            let mut var = 0.0;
            for s in 0..spatial {
                let diff = x_data[base + s].to_f32() - mean;
                var += diff * diff;
            }
            var /= spatial as f32;

            let inv_std = 1.0 / (var + eps).sqrt();
            let g = gamma_data[ch].to_f32();
            let bet = beta_data[ch].to_f32();

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
// Float Generic Backward
// ============================================================

pub fn instance_norm_backward<T: DType>(
    grad_output: &Tensor<T>,
    x: &Tensor<T>,
    gamma: &Tensor<T>,
    eps: f32,
) -> Vec<Tensor<T>> {
    let shape = x.shape();
    let c = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];

    let grad_data = grad_output.data();
    let x_data = x.data();
    let gamma_data = gamma.data();

    let mut grad_x = vec![T::from_f32(0.0); x.len()];

    for b in 0..batch {
        for ch in 0..c {
            let base = (b * c + ch) * spatial;

            let mut mean = 0.0;
            for s in 0..spatial {
                mean += x_data[base + s].to_f32();
            }
            mean /= spatial as f32;

            let mut var = 0.0;
            for s in 0..spatial {
                let diff = x_data[base + s].to_f32() - mean;
                var += diff * diff;
            }
            var /= spatial as f32;

            let inv_std = 1.0 / (var + eps).sqrt();
            let g = gamma_data[ch].to_f32();

            for s in 0..spatial {
                let idx = base + s;
                grad_x[idx] =
                    T::from_f32(grad_data[idx].to_f32() * g * inv_std);
            }
        }
    }

    vec![Tensor::new(grad_x, shape)]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct InstanceNormOp;

impl<T: DType + Send + Sync> Operator<T> for InstanceNormOp {
    fn name(&self) -> &'static str {
        "instance_norm"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 3);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        instance_norm(inputs[0], inputs[1], inputs[2], eps)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 3);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        instance_norm_backward(grad, inputs[0], inputs[1], eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_norm() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        );
        let gamma = Tensor::new(vec![1.0, 1.0], &[2]);
        let beta = Tensor::new(vec![0.0, 0.0], &[2]);
        let c = instance_norm(&x, &gamma, &beta, 1e-5);
        assert_eq!(c.shape(), &[2, 2, 2]);
        // Each (batch, channel) is normalized independently
    }
}
