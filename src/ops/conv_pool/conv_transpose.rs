// InferTrain - A Unified Inference and Training Engine
//
// Copyright (c) 2026 Jia Liu & InferTrain Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of InferTrain.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::dtype::DType;
use crate::ops::registry::{OpAttrs, Operator};
use crate::tensor::Tensor;

// ============================================================
// Correct transpose convolution output size calculation
// ============================================================

pub fn conv_transpose_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> usize {
    // Standard transpose convolution formula:
    // out = (in - 1) * stride - 2 * padding + kernel_size + output_padding
    (input_size - 1) * stride + kernel_size - 2 * padding + output_padding
}

// ============================================================
// Float Generic Forward
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
    let (out_c, _in_c_w, k_h, k_w) =
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    // Correct transpose convolution output size
    let out_h = (h - 1) * stride + k_h - 2 * padding + output_padding;
    let out_w = (w - 1) * stride + k_w - 2 * padding + output_padding;

    let out_size = n * out_c * out_h * out_w;
    let mut out_data = vec![T::from_f32(0.0); out_size];

    let x_data = x.data();
    let w_data = weight.data();
    let bias_data = bias.map(|b| b.data());

    // Transpose convolution implementation: for each input pixel,
    // multiply weight to corresponding output position
    for n_idx in 0..n {
        for oc in 0..out_c {
            for ic in 0..in_c {
                for ih in 0..h {
                    for iw in 0..w {
                        let x_val =
                            x_data[((n_idx * in_c + ic) * h + ih) * w + iw];

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let oh = ih * stride + kh - padding;
                                let ow = iw * stride + kw - padding;

                                if oh < out_h && ow < out_w {
                                    let w_idx = ((oc * in_c + ic) * k_h + kh)
                                        * k_w
                                        + kw;
                                    let out_idx =
                                        ((n_idx * out_c + oc) * out_h + oh)
                                            * out_w
                                            + ow;
                                    out_data[out_idx] = out_data[out_idx]
                                        + x_val * w_data[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if let Some(b) = bias_data {
        for n_idx in 0..n {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let out_idx =
                            ((n_idx * out_c + oc) * out_h + oh) * out_w + ow;
                        out_data[out_idx] = out_data[out_idx] + b[oc];
                    }
                }
            }
        }
    }

    Tensor::new(out_data, &[n, out_c, out_h, out_w])
}

// ============================================================
// Float Generic Backward - Simplified   TODO: Improve
// ============================================================

pub fn conv_transpose_backward<T: DType>(
    grad_output: &Tensor<T>,
) -> Vec<Tensor<T>> {
    vec![grad_output.clone()]
}

// ============================================================
// Operator Trait Implementation
// ============================================================

pub struct ConvTransposeOp;

impl<T: DType + Send + Sync> Operator<T> for ConvTransposeOp {
    fn name(&self) -> &'static str {
        "conv_transpose"
    }
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T> {
        assert!(inputs.len() >= 2 && inputs.len() <= 3);
        let stride = attrs.get_int("stride").map(|v| v as usize).unwrap_or(1);
        let padding = attrs.get_int("padding").map(|v| v as usize).unwrap_or(0);
        let output_padding =
            attrs.get_int("output_padding").map(|v| v as usize).unwrap_or(0);
        let bias = if inputs.len() == 3 { Some(inputs[2]) } else { None };
        conv_transpose(
            inputs[0],
            inputs[1],
            bias,
            stride,
            padding,
            output_padding,
        )
    }
    fn backward(
        &self,
        grad: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
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
        // Verify each pixel
        // x = [[1,2],[3,4]], w = [[1,0],[0,1]]
        // Output accumulates at corresponding positions
        assert_eq!(c.data()[0], 1.0); // (0,0) = 1*1
        assert_eq!(c.data()[1], 0.0); // (0,1) = 0
        assert_eq!(c.data()[4], 0.0); // (1,0) = 0
        assert_eq!(c.data()[5], 1.0); // (1,1) = 2*1 + 1*1? Actually 2 + 1 = 3?
                                      // Need to verify after reimplementation
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
        // All elements plus bias 1
        assert_eq!(c.data()[0], 2.0);
    }
}
