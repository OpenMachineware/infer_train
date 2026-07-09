// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// 辅助函数
// ============================================================

fn conv3d_output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let effective_kernel = (kernel - 1) * dilation + 1;
    (input + 2 * padding - effective_kernel) / stride + 1
}

// ============================================================
// 浮点泛型 Forward
// ============================================================

pub fn conv3d<T: DType + Send + Sync>(
    x: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor<T> {
    let x_shape = x.shape();
    let w_shape = weight.shape();

    assert_eq!(x_shape.len(), 5, "conv3d input must be 5D: [N, C, D, H, W]");
    assert_eq!(
        w_shape.len(),
        5,
        "conv3d weight must be 5D: [out_c, in_c, KD, KH, KW]"
    );

    let (n, in_c, d, h, w) =
        (x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]);
    let (out_c, in_c_w, k_d, k_h, k_w) =
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3], w_shape[4]);

    assert_eq!(in_c, in_c_w * groups);

    let out_d = conv3d_output_size(d, k_d, stride, padding, dilation);
    let out_h = conv3d_output_size(h, k_h, stride, padding, dilation);
    let out_w = conv3d_output_size(w, k_w, stride, padding, dilation);

    let out_size = n * out_c * out_d * out_h * out_w;
    let mut out_data = vec![T::from_f32(0.0); out_size];

    let x_data = x.data();
    let w_data = weight.data();
    let bias_data = bias.map(|b| b.data());

    let in_c_per_group = in_c / groups;
    let out_c_per_group = out_c / groups;

    for n_idx in 0..n {
        for g in 0..groups {
            for oc in 0..out_c_per_group {
                let out_ch = g * out_c_per_group + oc;
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = T::from_f32(0.0);
                            let d_start = od * stride;
                            let h_start = oh * stride;
                            let w_start = ow * stride;

                            for ic in 0..in_c_per_group {
                                let in_ch = g * in_c_per_group + ic;
                                for kd in 0..k_d {
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let id = d_start + kd * dilation;
                                            let ih = h_start + kh * dilation;
                                            let iw = w_start + kw * dilation;
                                            if id < d && ih < h && iw < w {
                                                let x_idx = (((n_idx * in_c
                                                    + in_ch)
                                                    * d
                                                    + id)
                                                    * h
                                                    + ih)
                                                    * w
                                                    + iw;
                                                let w_idx = (((out_ch
                                                    * in_c_per_group
                                                    + ic)
                                                    * k_d
                                                    + kd)
                                                    * k_h
                                                    + kh)
                                                    * k_w
                                                    + kw;
                                                sum = sum
                                                    + x_data[x_idx]
                                                        * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            let out_idx = (((n_idx * out_c + out_ch) * out_d
                                + od)
                                * out_h
                                + oh)
                                * out_w
                                + ow;
                            out_data[out_idx] = if let Some(b) = bias_data {
                                sum + b[out_ch]
                            } else {
                                sum
                            };
                        }
                    }
                }
            }
        }
    }

    Tensor::new(out_data, &[n, out_c, out_d, out_h, out_w])
}

// ============================================================
// 浮点泛型 Backward - 简化版   TODO: 完善
// ============================================================

pub fn conv3d_backward<T: DType>(grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait 实现
// ============================================================

pub struct Conv3dOp;

impl<T: DType + Send + Sync> Operator<T> for Conv3dOp {
    fn name(&self) -> &'static str {
        "conv3d"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").unwrap_or(1) as usize;
        let padding = attrs.get_int("padding").unwrap_or(0) as usize;
        let dilation = attrs.get_int("dilation").unwrap_or(1) as usize;
        let groups = attrs.get_int("groups").unwrap_or(1) as usize;
        let bias = if inputs.len() == 3 { Some(inputs[2]) } else { None };
        conv3d(inputs[0], inputs[1], bias, stride, padding, dilation, groups)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        conv3d_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv3d() {
        let x = Tensor::new(vec![1.0; 8], &[1, 1, 2, 2, 2]);
        let w = Tensor::new(vec![1.0; 8], &[1, 1, 2, 2, 2]);
        let c = conv3d(&x, &w, None, 1, 0, 1, 1);
        assert_eq!(c.shape(), &[1, 1, 1, 1, 1]);
    }
}
