// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn group_norm<T: DType + Send + Sync>(
    x: &Tensor<T>,
    gamma: &Tensor<T>,
    beta: &Tensor<T>,
    num_groups: usize,
    eps: f32,
) -> Tensor<T> {
    let shape = x.shape();
    assert!(shape.len() >= 2, "group_norm requires at least 2D tensor");

    let c = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];

    assert!(c % num_groups == 0, "channels must be divisible by num_groups");
    let group_size = c / num_groups;

    let x_data = x.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();

    let mut out_data = vec![T::from_f32(0.0); x.len()];

    for b in 0..batch {
        for g in 0..num_groups {
            let g_start = g * group_size;
            let g_end = g_start + group_size;

            // 计算该 group 的 mean 和 variance
            let mut mean = 0.0;
            let mut count = 0;
            for ch in g_start..g_end {
                let base = (b * c + ch) * spatial;
                for s in 0..spatial {
                    mean += x_data[base + s].to_f32();
                    count += 1;
                }
            }
            mean /= count as f32;

            let mut var = 0.0;
            for ch in g_start..g_end {
                let base = (b * c + ch) * spatial;
                for s in 0..spatial {
                    let diff = x_data[base + s].to_f32() - mean;
                    var += diff * diff;
                }
            }
            var /= count as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            // 应用归一化
            for ch in g_start..g_end {
                let base = (b * c + ch) * spatial;
                let g_idx = ch;
                let gamma_val = gamma_data[g_idx].to_f32();
                let beta_val = beta_data[g_idx].to_f32();
                for s in 0..spatial {
                    let idx = base + s;
                    let norm = (x_data[idx].to_f32() - mean) * inv_std;
                    out_data[idx] = T::from_f32(gamma_val * norm + beta_val);
                }
            }
        }
    }

    Tensor::new(out_data, shape)
}

// ============================================================
// 2. 浮点泛型 Backward (简化版)
// ============================================================

pub fn group_norm_backward<T: DType>(
    grad_output: &Tensor<T>,
    x: &Tensor<T>,
    gamma: &Tensor<T>,
    num_groups: usize,
    eps: f32,
) -> Vec<Tensor<T>> {
    let shape = x.shape();
    let c = shape[1];
    let spatial: usize = shape[2..].iter().product();
    let batch = shape[0];
    let group_size = c / num_groups;

    let grad_data = grad_output.data();
    let x_data = x.data();
    let gamma_data = gamma.data();

    let mut grad_x = vec![T::from_f32(0.0); x.len()];

    for b in 0..batch {
        for g in 0..num_groups {
            let g_start = g * group_size;
            let g_end = g_start + group_size;

            let mut mean = 0.0;
            let mut count = 0;
            for ch in g_start..g_end {
                let base = (b * c + ch) * spatial;
                for s in 0..spatial {
                    mean += x_data[base + s].to_f32();
                    count += 1;
                }
            }
            mean /= count as f32;

            let mut var = 0.0;
            for ch in g_start..g_end {
                let base = (b * c + ch) * spatial;
                for s in 0..spatial {
                    let diff = x_data[base + s].to_f32() - mean;
                    var += diff * diff;
                }
            }
            var /= count as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            for ch in g_start..g_end {
                let base = (b * c + ch) * spatial;
                let gamma_val = gamma_data[ch].to_f32();
                for s in 0..spatial {
                    let idx = base + s;
                    grad_x[idx] = T::from_f32(grad_data[idx].to_f32() * gamma_val * inv_std);
                }
            }
        }
    }

    vec![Tensor::new(grad_x, shape)]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct GroupNormOp;

impl<T: DType + Send + Sync> Operator<T> for GroupNormOp {
    fn name(&self) -> &'static str { "group_norm" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert_eq!(inputs.len(), 3);
        let num_groups = attrs.get_int("num_groups").unwrap_or(1) as usize;
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        group_norm(inputs[0], inputs[1], inputs[2], num_groups, eps)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        assert_eq!(inputs.len(), 3);
        let num_groups = attrs.get_int("num_groups").unwrap_or(1) as usize;
        let eps = attrs.get_float("eps").unwrap_or(1e-5);
        group_norm_backward(grad, inputs[0], inputs[1], num_groups, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_norm() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4, 1]);
        let gamma = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], &[4]);
        let beta = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], &[4]);
        let c = group_norm(&x, &gamma, &beta, 2, 1e-5);
        assert_eq!(c.shape(), &[2, 4, 1]);
        // 每个 group 独立归一化
    }
}
