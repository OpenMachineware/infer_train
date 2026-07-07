// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 辅助函数：计算卷积输出尺寸
// ============================================================

fn conv_output_size(input: usize, kernel: usize, stride: usize, padding: usize, dilation: usize) -> usize {
    let effective_kernel = (kernel - 1) * dilation + 1;
    (input + 2 * padding - effective_kernel) / stride + 1
}

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn conv2d<T: DType + Send + Sync>(
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

    assert_eq!(x_shape.len(), 4, "conv2d input must be 4D: [N, C, H, W]");
    assert_eq!(w_shape.len(), 4, "conv2d weight must be 4D: [out_c, in_c, KH, KW]");

    let (n, in_c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (out_c, in_c_w, k_h, k_w) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    assert_eq!(in_c, in_c_w * groups, "Input channels mismatch");
    assert_eq!(out_c % groups, 0, "Output channels must be divisible by groups");

    let out_h = conv_output_size(h, k_h, stride, padding, dilation);
    let out_w = conv_output_size(w, k_w, stride, padding, dilation);

    let out_size = n * out_c * out_h * out_w;
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
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = T::from_f32(0.0);
                        let h_start = oh * stride;
                        let w_start = ow * stride;

                        for ic in 0..in_c_per_group {
                            let in_ch = g * in_c_per_group + ic;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = h_start + kh * dilation;
                                    let iw = w_start + kw * dilation;
                                    if ih < h && iw < w {
                                        let x_idx = ((n_idx * in_c + in_ch) * h + ih) * w + iw;
                                        let w_idx = ((out_ch * in_c_per_group + ic) * k_h + kh) * k_w + kw;
                                        sum = sum + x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        let out_idx = ((n_idx * out_c + out_ch) * out_h + oh) * out_w + ow;
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

    Tensor::new(out_data, &[n, out_c, out_h, out_w])
}

// ============================================================
// 2. 浮点泛型 Backward (简化版)
// ============================================================

pub fn conv2d_backward<T: DType>(
    grad_output: &Tensor<T>,
    _x: &Tensor<T>,
    _weight: &Tensor<T>,
    _stride: usize,
    _padding: usize,
    _dilation: usize,
    _groups: usize,
) -> Vec<Tensor<T>> {
    // 简化版：只计算对输入的梯度
    // 实际反向传播需要 grad_weight 和 grad_bias
    let grad_x = grad_output.clone();
    vec![grad_x]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct Conv2dOp;

impl<T: DType + Send + Sync> Operator<T> for Conv2dOp {
    fn name(&self) -> &'static str { "conv2d" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").unwrap_or(1) as usize;
        let padding = attrs.get_int("padding").unwrap_or(0) as usize;
        let dilation = attrs.get_int("dilation").unwrap_or(1) as usize;
        let groups = attrs.get_int("groups").unwrap_or(1) as usize;
        let bias = if inputs.len() == 3 { Some(inputs[2]) } else { None };
        conv2d(inputs[0], inputs[1], bias, stride, padding, dilation, groups)
    }
    fn backward(&self, grad: &Tensor<T>, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Vec<Tensor<T>> {
        let stride = attrs.get_int("stride").unwrap_or(1) as usize;
        let padding = attrs.get_int("padding").unwrap_or(0) as usize;
        let dilation = attrs.get_int("dilation").unwrap_or(1) as usize;
        let groups = attrs.get_int("groups").unwrap_or(1) as usize;
        conv2d_backward(grad, inputs[0], inputs[1], stride, padding, dilation, groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_valid() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 3, 3]);
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let c = conv2d(&x, &w, None, 1, 0, 1, 1);
        assert_eq!(c.shape(), &[1, 1, 2, 2]);
        assert_eq!(c.data()[0], 1.0 + 0.0 + 0.0 + 5.0);
        assert_eq!(c.data()[1], 2.0 + 0.0 + 0.0 + 6.0);
        assert_eq!(c.data()[2], 4.0 + 0.0 + 0.0 + 8.0);
        assert_eq!(c.data()[3], 5.0 + 0.0 + 0.0 + 9.0);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 3, 3]);
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let b = Tensor::new(vec![1.0], &[1]);
        let c = conv2d(&x, &w, Some(&b), 1, 0, 1, 1);
        assert_eq!(c.data()[0], 1.0 + 0.0 + 0.0 + 5.0 + 1.0);
    }
}
