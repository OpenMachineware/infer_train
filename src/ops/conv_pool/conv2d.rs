// use rayon::prelude::*;
use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Helper: Compute convolution output size
// ============================================================

fn conv_output_size(
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
// Helper: ConvTranspose (Deconvolution)
// ============================================================

fn conv_transpose_impl<T: DType + Send + Sync>(
    grad_output: &Tensor<T>,
    weight: &Tensor<T>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Tensor<T> {
    let w_shape = weight.shape();
    let out_c = w_shape[0];
    let in_c = w_shape[1];
    let k_h = w_shape[2];
    let k_w = w_shape[3];

    let g_shape = grad_output.shape();
    let n = g_shape[0];
    let h = g_shape[2];
    let w = g_shape[3];

    let out_h = (h - 1) * stride + k_h - 2 * padding + output_padding;
    let out_w = (w - 1) * stride + k_w - 2 * padding + output_padding;

    let mut result = vec![T::zero(); n * in_c * out_h * out_w];
    let grad_data = grad_output.data();
    let weight_data = weight.data();

    for b in 0..n {
        for oc in 0..out_c {
            for ic in 0..in_c {
                for i in 0..h {
                    for j in 0..w {
                        let grad_val =
                            grad_data[((b * out_c + oc) * h + i) * w + j];
                        if grad_val == T::zero() {
                            continue;
                        }

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let oh = i * stride + kh - padding;
                                let ow = j * stride + kw - padding;
                                if oh < out_h && ow < out_w {
                                    let w_idx = ((oc * in_c + ic) * k_h + kh)
                                        * k_w
                                        + kw;
                                    let out_idx =
                                        ((b * in_c + ic) * out_h + oh) * out_w
                                            + ow;
                                    result[out_idx] = result[out_idx]
                                        + grad_val * weight_data[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::new(result, &[n, in_c, out_h, out_w])
}

// ============================================================
// Helper: Weight Gradient
// ============================================================

fn conv_weight_gradient_impl<T: DType + Send + Sync>(
    input: &Tensor<T>,
    grad_output: &Tensor<T>,
    stride: usize,
    _padding: usize,
    _dilation: usize,
    _groups: usize,
) -> Tensor<T> {
    let i_shape = input.shape();
    let g_shape = grad_output.shape();
    let in_c = i_shape[1];
    let out_c = g_shape[1];
    let h = i_shape[2];
    let w = i_shape[3];
    let gh = g_shape[2];
    let gw = g_shape[3];

    let k_h = h - (gh - 1) * stride;
    let k_w = w - (gw - 1) * stride;

    let mut grad_weight = vec![T::zero(); out_c * in_c * k_h * k_w];
    let input_data = input.data();
    let grad_data = grad_output.data();

    for b in 0..i_shape[0] {
        for oc in 0..out_c {
            for ic in 0..in_c {
                for i in 0..gh {
                    for j in 0..gw {
                        let grad_val =
                            grad_data[((b * out_c + oc) * gh + i) * gw + j];
                        if grad_val == T::zero() {
                            continue;
                        }

                        let h_start = i * stride;
                        let w_start = j * stride;
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = h_start + kh;
                                let iw = w_start + kw;
                                if ih < h && iw < w {
                                    let input_idx =
                                        ((b * in_c + ic) * h + ih) * w + iw;
                                    let w_idx = ((oc * in_c + ic) * k_h + kh)
                                        * k_w
                                        + kw;
                                    grad_weight[w_idx] = grad_weight[w_idx]
                                        + grad_val * input_data[input_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::new(grad_weight, &[out_c, in_c, k_h, k_w])
}

// ============================================================
// Helper: Bias Gradient
// ============================================================

fn conv_bias_gradient_impl<T: DType + Send + Sync>(
    grad_output: &Tensor<T>,
) -> Tensor<T> {
    let shape = grad_output.shape();
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    let grad_data = grad_output.data();

    let mut bias_grad = vec![T::zero(); c];
    for b in 0..n {
        for ch in 0..c {
            let mut sum = T::zero();
            for i in 0..h {
                for j in 0..w {
                    let idx = ((b * c + ch) * h + i) * w + j;
                    sum = sum + grad_data[idx];
                }
            }
            bias_grad[ch] = bias_grad[ch] + sum;
        }
    }

    Tensor::new(bias_grad, &[c])
}

// ============================================================
// Float Generic Forward
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
    assert_eq!(
        w_shape.len(),
        4,
        "conv2d weight must be 4D: [out_c, in_c, KH, KW]"
    );

    let (n, in_c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (out_c, in_c_w, k_h, k_w) =
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    assert_eq!(in_c, in_c_w * groups, "Input channels mismatch");
    assert_eq!(
        out_c % groups,
        0,
        "Output channels must be divisible by groups"
    );

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
                                        let x_idx =
                                            ((n_idx * in_c + in_ch) * h + ih)
                                                * w
                                                + iw;
                                        let w_idx = ((out_ch * in_c_per_group
                                            + ic)
                                            * k_h
                                            + kh)
                                            * k_w
                                            + kw;
                                        sum =
                                            sum + x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        let out_idx = ((n_idx * out_c + out_ch) * out_h + oh)
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

    Tensor::new(out_data, &[n, out_c, out_h, out_w])
}

// ============================================================
// Float Generic Backward
// ============================================================

pub fn conv2d_backward<T: DType + Send + Sync>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Vec<Tensor<T>> {
    // ∂L/∂input = conv_transpose(grad_output, weight)
    let grad_input =
        conv_transpose_impl(grad_output, weight, stride, padding, 0);

    // ∂L/∂weight = conv(input, grad_output)
    let grad_weight = conv_weight_gradient_impl(
        input,
        grad_output,
        stride,
        padding,
        dilation,
        groups,
    );

    // ∂L/∂bias = sum(grad_output) over [N, H, W]
    let grad_bias = conv_bias_gradient_impl(grad_output);

    vec![grad_input, grad_weight, grad_bias]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct Conv2dOp;

impl<T: DType + Send + Sync> Operator<T> for Conv2dOp {
    fn name(&self) -> &'static str {
        "conv2d"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").map(|v| v as usize).unwrap_or(1);
        let padding = attrs.get_int("padding").map(|v| v as usize).unwrap_or(0);
        let dilation =
            attrs.get_int("dilation").map(|v| v as usize).unwrap_or(1);
        let groups = attrs.get_int("groups").map(|v| v as usize).unwrap_or(1);
        let bias = if inputs.len() == 3 { Some(inputs[2]) } else { None };
        conv2d(inputs[0], inputs[1], bias, stride, padding, dilation, groups)
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        inputs: &[&Tensor<T>],
        attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").map(|v| v as usize).unwrap_or(1);
        let padding = attrs.get_int("padding").map(|v| v as usize).unwrap_or(0);
        let dilation =
            attrs.get_int("dilation").map(|v| v as usize).unwrap_or(1);
        let groups = attrs.get_int("groups").map(|v| v as usize).unwrap_or(1);
        conv2d_backward(
            grad, inputs[0], inputs[1], stride, padding, dilation, groups,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_valid() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
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
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        );
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let b = Tensor::new(vec![1.0], &[1]);
        let c = conv2d(&x, &w, Some(&b), 1, 0, 1, 1);
        assert_eq!(c.data()[0], 1.0 + 0.0 + 0.0 + 5.0 + 1.0);
    }
}
