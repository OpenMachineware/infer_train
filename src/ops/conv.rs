// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! CNN operators: CONV_2D, CONV_2D_DW, POOL_2D, IM2COL

#[cfg(target_arch = "aarch64")]
use super::arm::neon;

/// 2D Convolution (simple, no padding)
/// Optimized: input_channel major order for better cache reuse
pub fn conv_2d_f32_simple(
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
) {
    let h_out = h - kh + 1;
    let w_out = w - kw + 1;

    // Zero output
    output[..c_out * h_out * w_out].fill(0.0);

    // Process by input channel for better input cache reuse
    for ic in 0..c_in {
        let ic_offset = ic * h * w;

        for oc in 0..c_out {
            let w_ic_offset = oc * c_in * kh * kw + ic * kh * kw;
            let out_ch_base = oc * h_out * w_out;

            for oh in 0..h_out {
                let out_row_base = out_ch_base + oh * w_out;
                let in_row_base = ic_offset + oh * w;

                for ow in 0..w_out {
                    let mut sum =
                        unsafe { *output.get_unchecked(out_row_base + ow) };

                    if kh == 3 && kw == 3 {
                        // Row 0
                        let in_base = in_row_base + ow;
                        sum += unsafe {
                            *input.get_unchecked(in_base)
                                * *weights.get_unchecked(w_ic_offset)
                        };
                        sum += unsafe {
                            *input.get_unchecked(in_base + 1)
                                * *weights.get_unchecked(w_ic_offset + 1)
                        };
                        sum += unsafe {
                            *input.get_unchecked(in_base + 2)
                                * *weights.get_unchecked(w_ic_offset + 2)
                        };

                        // Row 1
                        let in_r1 = in_base + w;
                        sum += unsafe {
                            *input.get_unchecked(in_r1)
                                * *weights.get_unchecked(w_ic_offset + 3)
                        };
                        sum += unsafe {
                            *input.get_unchecked(in_r1 + 1)
                                * *weights.get_unchecked(w_ic_offset + 4)
                        };
                        sum += unsafe {
                            *input.get_unchecked(in_r1 + 2)
                                * *weights.get_unchecked(w_ic_offset + 5)
                        };

                        // Row 2
                        let in_r2 = in_base + 2 * w;
                        sum += unsafe {
                            *input.get_unchecked(in_r2)
                                * *weights.get_unchecked(w_ic_offset + 6)
                        };
                        sum += unsafe {
                            *input.get_unchecked(in_r2 + 1)
                                * *weights.get_unchecked(w_ic_offset + 7)
                        };
                        sum += unsafe {
                            *input.get_unchecked(in_r2 + 2)
                                * *weights.get_unchecked(w_ic_offset + 8)
                        };
                    } else {
                        for kiy in 0..kh {
                            for kix in 0..kw {
                                let in_idx = in_row_base + kiy * w + ow + kix;
                                let w_idx = w_ic_offset + kiy * kw + kix;
                                sum += unsafe {
                                    *input.get_unchecked(in_idx)
                                        * *weights.get_unchecked(w_idx)
                                };
                            }
                        }
                    }

                    unsafe {
                        *output.get_unchecked_mut(out_row_base + ow) = sum
                    };
                }
            }
        }
    }
}

/// 2D Convolution
/// input: [C_in, H, W] input tensor
/// weights: [C_out, C_in, kH, kW] convolution kernels
/// bias: [C_out] optional bias
/// output: [C_out, H_out, W_out] output tensor
pub fn conv_2d_f32(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) {
    // Use simple version for common case (stride=1, padding=0)
    if stride == 1 && padding == 0 {
        conv_2d_f32_simple(input, weights, output, c_in, c_out, h, w, kh, kw);
        return;
    }

    // Use im2col + GEMM for general case
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let patch_n = h_out * w_out;
    let knl_n = c_in * kh * kw;

    // Allocate im2col buffer
    let mut patches = vec![0.0f32; patch_n * knl_n];

    // im2col: convert input to patches matrix
    for oh in 0..h_out {
        for ow in 0..w_out {
            let patch_idx = oh * w_out + ow;
            let patch_base = patch_idx * knl_n;

            for ic in 0..c_in {
                let ic_offset = ic * kh * kw;
                let in_ch_base = ic * h * w;

                for kiy in 0..kh {
                    let iy = (oh * stride + kiy) as isize - padding as isize;

                    for kix in 0..kw {
                        let ix =
                            (ow * stride + kix) as isize - padding as isize;

                        let knl_idx = ic_offset + kiy * kw + kix;
                        let val = if iy >= 0
                            && ix >= 0
                            && (iy as usize) < h
                            && (ix as usize) < w
                        {
                            let in_idx =
                                in_ch_base + (iy as usize) * w + (ix as usize);
                            input[in_idx]
                        } else {
                            0.0
                        };

                        patches[patch_base + knl_idx] = val;
                    }
                }
            }
        }
    }

    // GEMM: output = patches × weights^T
    // patches: [patch_n, knl_n]
    // weights: [c_out, knl_n] (stored as [OC, IC, KH, KW])
    // output: [patch_n, c_out]
    unsafe {
        crate::quant::matmul::fp32::sgemm(
            &patches, weights, output, c_out,   // m = c_out
            patch_n, // n = patch_n
            knl_n,   // k = knl_n
        );
    }

    // Add bias
    if let Some(bias) = bias {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let patch_idx = oh * w_out + ow;
                for oc in 0..c_out {
                    let out_idx = oc * h_out * w_out + patch_idx;
                    output[out_idx] += bias[oc];
                }
            }
        }
    }
}

/// 2D Convolution with dispatch (NEON on aarch64, scalar otherwise)
pub fn conv_2d_f32_dispatch(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::conv_2d_f32(
                input, weights, bias, output, c_in, c_out, h, w, kh, kw,
                stride, padding,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        conv_2d_f32(
            input, weights, bias, output, c_in, c_out, h, w, kh, kw, stride,
            padding,
        )
    }
}

/// Depthwise 2D Convolution
/// Each input channel has its own filter
/// input: [C, H, W]
/// weights: [C, kH, kW]
/// output: [C, H_out, W_out]
pub fn conv_2d_dw_f32(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) {
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;

    for ch in 0..c {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = bias.map(|b| b[ch]).unwrap_or(0.0);

                for kiy in 0..kh {
                    for kix in 0..kw {
                        let iy = oh * stride + kiy;
                        let ix = ow * stride + kix;

                        let iy = iy as isize - padding as isize;
                        let ix = ix as isize - padding as isize;

                        if iy >= 0
                            && ix >= 0
                            && (iy as usize) < h
                            && (ix as usize) < w
                        {
                            let in_idx =
                                ch * h * w + (iy as usize) * w + (ix as usize);
                            let w_idx = ch * kh * kw + kiy * kw + kix;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }

                let out_idx = ch * h_out * w_out + oh * w_out + ow;
                output[out_idx] = sum;
            }
        }
    }
}

/// 2D Max Pooling - simple scalar implementation
/// Optimized to match C++ codegen: uses iterator for tight inner loop
#[inline]
pub fn pool_2d_max_f32(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    let h_out = (h - kh) / stride + 1;
    let w_out = (w - kw) / stride + 1;

    // Specialized kernels for small sizes (NEON on aarch64)
    #[cfg(target_arch = "aarch64")]
    if stride == 2 {
        if kh == 2 && kw == 2 {
            return unsafe { neon::pool_2x2(input, output, c, h, w) };
        }
        if kh == 3 && kw == 3 {
            return unsafe { neon::pool_3x3(input, output, c, h, w) };
        }
    }

    // Generic fallback
    for ch in 0..c {
        let ch_offset = ch * h * w;
        let out_offset = ch * h_out * w_out;

        for oh in 0..h_out {
            let iy_start = oh * stride;

            for ow in 0..w_out {
                let ix = ow * stride;
                let mut max_val = f32::NEG_INFINITY;

                let mut kiy = 0;
                while kiy < kh {
                    let iy = iy_start + kiy;
                    let row_offset = ch_offset + iy * w + ix;

                    let row = &input[row_offset..row_offset + kw];
                    for &val in row {
                        max_val = max_val.max(val);
                    }
                    kiy += 1;
                }

                output[out_offset + oh * w_out + ow] = max_val;
            }
        }
    }
}

/// 2D Average Pooling
pub fn pool_2d_avg_f32(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    let h_out = (h - kh) / stride + 1;
    let w_out = (w - kw) / stride + 1;
    let k_size = kh * kw;
    let scale = 1.0 / k_size as f32;

    for ch in 0..c {
        let ch_offset = ch * h * w;
        let out_offset = ch * h_out * w_out;

        for oh in 0..h_out {
            let iy_start = oh * stride;

            for ow in 0..w_out {
                let ix_start = ow * stride;
                let mut sum = 0.0f32;

                for kiy in 0..kh {
                    let iy = iy_start + kiy;
                    let row_offset = ch_offset + iy * w + ix_start;

                    for kix in 0..kw {
                        sum += input[row_offset + kix];
                    }
                }

                output[out_offset + oh * w_out + ow] = sum * scale;
            }
        }
    }
}

/// 2D Average Pooling with dispatch (NEON on aarch64, scalar otherwise)
pub fn pool_2d_avg_f32_dispatch(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::pool_2d_avg_f32(input, output, c, h, w, kh, kw, stride) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        pool_2d_avg_f32(input, output, c, h, w, kh, kw, stride)
    }
}

/// Im2col: Convert image patches to columns for efficient GEMM
/// Used to implement convolution as matrix multiplication
pub fn im2col_f32(
    input: &[f32],
    output: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) {
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;

    // Output shape: [h_out * w_out, c_in * kh * kw]
    for oh in 0..h_out {
        for ow in 0..w_out {
            let col_idx = (oh * w_out + ow) * c_in * kh * kw;

            let mut pos = 0;
            for ic in 0..c_in {
                for kiy in 0..kh {
                    for kix in 0..kw {
                        let iy = oh * stride + kiy;
                        let ix = ow * stride + kix;

                        let iy = iy as isize - padding as isize;
                        let ix = ix as isize - padding as isize;

                        if iy >= 0
                            && ix >= 0
                            && (iy as usize) < h
                            && (ix as usize) < w
                        {
                            let in_idx =
                                ic * h * w + (iy as usize) * w + (ix as usize);
                            output[col_idx + pos] = input[in_idx];
                        } else {
                            output[col_idx + pos] = 0.0; // Zero padding
                        }
                        pos += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_2d() {
        // 1x3x3 input, 1x1x3x3 kernel, stride 1, no padding
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weights = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // identity
        let mut output = [0.0f32; 1];

        conv_2d_f32(
            &input,
            &weights,
            None,
            &mut output,
            1,
            1,
            3,
            3,
            3,
            3,
            1,
            0,
        );

        // Single output: sum of diagonal elements
        assert!((output[0] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_pool_2d() {
        // 1x4x4 input, 2x2 max pool
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut output = [0.0f32; 4];

        pool_2d_max_f32(&input, &mut output, 1, 4, 4, 2, 2, 2);

        assert_eq!(output, [6.0, 8.0, 14.0, 16.0]);
    }
}
