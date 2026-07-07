// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn conv1d<T: DType + Send + Sync>(
    x: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    _groups: usize,
) -> Tensor<T> {
    let x_shape = x.shape();
    let w_shape = weight.shape();

    assert_eq!(x_shape.len(), 3, "conv1d input must be 3D: [N, C, L]");
    assert_eq!(w_shape.len(), 3, "conv1d weight must be 3D: [out_c, in_c, KL]");

    let (n, in_c, l) = (x_shape[0], x_shape[1], x_shape[2]);
    let (out_c, _in_c_w, k_l) = (w_shape[0], w_shape[1], w_shape[2]);

    let out_l = (l + 2 * padding - dilation * (k_l - 1) - 1) / stride + 1;

    let out_size = n * out_c * out_l;
    let mut out_data = vec![T::from_f32(0.0); out_size];

    let x_data = x.data();
    let w_data = weight.data();
    let bias_data = bias.map(|b| b.data());

    for n_idx in 0..n {
        for oc in 0..out_c {
            for ol in 0..out_l {
                let mut sum = T::from_f32(0.0);
                let l_start = ol * stride;
                for ic in 0..in_c {
                    for kl in 0..k_l {
                        let il = l_start + kl * dilation;
                        if il < l {
                            let x_idx = (n_idx * in_c + ic) * l + il;
                            let w_idx = (oc * in_c + ic) * k_l + kl;
                            sum = sum + x_data[x_idx] * w_data[w_idx];
                        }
                    }
                }
                let out_idx = (n_idx * out_c + oc) * out_l + ol;
                out_data[out_idx] = if let Some(b) = bias_data {
                    sum + b[oc]
                } else {
                    sum
                };
            }
        }
    }

    Tensor::new(out_data, &[n, out_c, out_l])
}

// ============================================================
// 2. 浮点泛型 Backward (简化版)
// ============================================================

pub fn conv1d_backward<T: DType>(
    grad_output: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct Conv1dOp;

impl<T: DType + Send + Sync> Operator<T> for Conv1dOp {
    fn name(&self) -> &'static str { "conv1d" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").unwrap_or(1) as usize;
        let padding = attrs.get_int("padding").unwrap_or(0) as usize;
        let dilation = attrs.get_int("dilation").unwrap_or(1) as usize;
        let groups = attrs.get_int("groups").unwrap_or(1) as usize;
        let bias = if inputs.len() == 3 { Some(inputs[2]) } else { None };
        conv1d(inputs[0], inputs[1], bias, stride, padding, dilation, groups)
    }
    fn backward(&self, grad: &Tensor<T>, _inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        conv1d_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let w = Tensor::new(vec![1.0, 2.0], &[1, 1, 2]);
        let c = conv1d(&x, &w, None, 1, 0, 1, 1);
        assert_eq!(c.shape(), &[1, 1, 4]);
    }
}
