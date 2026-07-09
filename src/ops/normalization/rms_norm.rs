// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Float Generic Forward
// ============================================================

pub fn rms_norm<T: DType + Send + Sync>(
    x: &Tensor<T>,
    weight: &Tensor<T>,
    eps: f32,
) -> Tensor<T> {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let outer: usize = shape[..shape.len() - 1].iter().product();

    let x_data = x.data();
    let weight_data = weight.data();

    let mut out_data = vec![T::from_f32(0.0); x.len()];

    for o in 0..outer {
        let base = o * last_dim;

        // Compute RMS
        let mut rms = 0.0;
        for i in 0..last_dim {
            let v = x_data[base + i].to_f32();
            rms += v * v;
        }
        rms = (rms / last_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..last_dim {
            let idx = base + i;
            out_data[idx] = T::from_f32(
                x_data[idx].to_f32() * inv_rms * weight_data[i].to_f32(),
            );
        }
    }

    Tensor::new(out_data, shape)
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn rms_norm_backward<T: DType + Send + Sync>(
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
    let weight_data = weight.data();

    let mut grad_input = vec![T::zero(); input.len()];
    let mut grad_weight = vec![T::zero(); last_dim];

    for o in 0..outer {
        let base = o * last_dim;

        let mut rms = 0.0;
        for i in 0..last_dim {
            let v = x_data[base + i].to_f32();
            rms += v * v;
        }
        rms = (rms / last_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..last_dim {
            let idx = base + i;
            let x = x_data[idx].to_f32();
            let grad = grad_data[idx].to_f32();
            let w = weight_data[i].to_f32();

            grad_input[idx] = T::from_f32(grad * w * inv_rms);
            grad_weight[i] =
                T::from_f32(grad_weight[i].to_f32() + grad * x * inv_rms);
        }
    }

    vec![Tensor::new(grad_input, shape), Tensor::new(grad_weight, &[last_dim])]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct RmsNormOp;

impl<T: DType + Send + Sync> Operator<T> for RmsNormOp {
    fn name(&self) -> &'static str {
        "rms_norm"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 2);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        rms_norm(inputs[0], inputs[1], eps)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 2);
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        rms_norm_backward(grad, inputs[0], inputs[1], eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], &[3]);
        let c = rms_norm(&x, &weight, 1e-5);
        assert_eq!(c.shape(), &[2, 3]);
        // After RMS normalization, each element should have RMS = 1
    }
}
