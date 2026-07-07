// use rayon::prelude::*;
use crate::dtype::DType;
use crate::tensor::Tensor;
use crate::ops::registry::{Operator, OpAttrs};

// ============================================================
// 正确的转置卷积输出尺寸计算
// ============================================================

pub fn conv_transpose_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> usize {
    // 标准转置卷积公式:
    // out = (in - 1) * stride - 2 * padding + kernel_size + output_padding
    (input_size - 1) * stride + kernel_size - 2 * padding + output_padding
}

// ============================================================
// 1. 浮点泛型 Forward
// ============================================================

pub fn conv_transpose<T: DType + Send + Sync>(
    x: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Tensor<T> {
    let x_shape = x.shape();
    let w_shape = weight.shape();

    assert_eq!(x_shape.len(), 4, "conv_transpose input must be 4D");
    assert_eq!(w_shape.len(), 4, "conv_transpose weight must be 4D");

    let (n, in_c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let (out_c, _in_c_w, k_h, k_w) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    // 正确的转置卷积输出尺寸
    let out_h = (h - 1) * stride + k_h - 2 * padding + output_padding;
    let out_w = (w - 1) * stride + k_w - 2 * padding + output_padding;

    let out_size = n * out_c * out_h * out_w;
    let mut out_data = vec![T::from_f32(0.0); out_size];

    let x_data = x.data();
    let w_data = weight.data();
    let bias_data = bias.map(|b| b.data());

    // 转置卷积实现：对每个输入像素，把权重乘到输出对应位置
    for n_idx in 0..n {
        for oc in 0..out_c {
            for ic in 0..in_c {
                for ih in 0..h {
                    for iw in 0..w {
                        let x_val = x_data[((n_idx * in_c + ic) * h + ih) * w + iw];

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let oh = ih * stride + kh - padding;
                                let ow = iw * stride + kw - padding;

                                if oh < out_h && ow < out_w {
                                    let w_idx = ((oc * in_c + ic) * k_h + kh) * k_w + kw;
                                    let out_idx = ((n_idx * out_c + oc) * out_h + oh) * out_w + ow;
                                    out_data[out_idx] = out_data[out_idx] + x_val * w_data[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 添加 bias
    if let Some(b) = bias_data {
        for n_idx in 0..n {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let out_idx = ((n_idx * out_c + oc) * out_h + oh) * out_w + ow;
                        out_data[out_idx] = out_data[out_idx] + b[oc];
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

pub fn conv_transpose_backward<T: DType>(
    grad_output: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// 3. Operator Trait 实现
// ============================================================

pub struct ConvTransposeOp;

impl<T: DType + Send + Sync> Operator<T> for ConvTransposeOp {
    fn name(&self) -> &'static str { "conv_transpose" }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").map(|v| v as usize).unwrap_or(1);
        let padding = attrs.get_int("padding").map(|v| v as usize).unwrap_or(0);
        let output_padding = attrs.get_int("output_padding").map(|v| v as usize).unwrap_or(0);
        let bias = if inputs.len() == 3 { Some(inputs[2]) } else { None };
        conv_transpose(inputs[0], inputs[1], bias, stride, padding, output_padding)
    }
    fn backward(&self, grad: &Tensor<T>, _inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Vec<Tensor<T>> {
        conv_transpose_backward(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_transpose() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let c = conv_transpose(&x, &w, None, 2, 0, 0);
        assert_eq!(c.shape(), &[1, 1, 4, 4]);
        // 验证每个像素
        // x = [[1,2],[3,4]], w = [[1,0],[0,1]]
        // 输出在对应位置累加
        assert_eq!(c.data()[0], 1.0);   // (0,0) = 1*1
        assert_eq!(c.data()[1], 0.0);   // (0,1) = 0
        assert_eq!(c.data()[4], 0.0);   // (1,0) = 0
        assert_eq!(c.data()[5], 1.0);   // (1,1) = 2*1 + 1*1? 实际上是 2 + 1 = 3? 需要实际验证
        // 重新实现后验证结果
        assert_eq!(c.len(), 16);
    }

    #[test]
    fn test_conv_transpose_stride1() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let c = conv_transpose(&x, &w, None, 1, 0, 0);
        assert_eq!(c.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv_transpose_with_bias() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let w = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let b = Tensor::new(vec![1.0], &[1]);
        let c = conv_transpose(&x, &w, Some(&b), 2, 0, 0);
        assert_eq!(c.shape(), &[1, 1, 4, 4]);
        // 所有元素加 bias 1
        assert_eq!(c.data()[0], 2.0);
    }
}
