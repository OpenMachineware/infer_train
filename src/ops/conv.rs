//! CNN operators: CONV_2D, CONV_2D_DW, POOL_2D, IM2COL

use std::arch::aarch64::*;

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

/// 2D Convolution with NEON optimization (1x1 and 3x3 kernels)
#[target_feature(enable = "neon")]
pub unsafe fn conv_2d_f32_neon(
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
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let kernel_size = kh * kw;

    for oc in 0..c_out {
        let w_oc_offset = oc * c_in * kernel_size;
        let out_oc_offset = oc * h_out * w_out;
        let bias_val = bias.map(|b| b[oc]).unwrap_or(0.0);

        for oh in 0..h_out {
            let out_row_offset = out_oc_offset + oh * w_out;

            for ow in 0..w_out {
                let mut sum = vdupq_n_f32(bias_val);

                for ic in 0..c_in {
                    let w_ic_offset = w_oc_offset + ic * kernel_size;
                    let in_ch_offset = ic * h * w;

                    for kiy in 0..kh {
                        let iy =
                            (oh * stride + kiy) as isize - padding as isize;
                        if iy < 0 || (iy as usize) >= h {
                            continue;
                        }

                        let row_offset = in_ch_offset + (iy as usize) * w;

                        // Process kw elements with NEON if possible
                        if kw >= 4 && padding == 0 {
                            let ix_start = ow * stride;
                            for kix in (0..kw).step_by(4) {
                                let ix = ix_start + kix;
                                if ix + 3 < w {
                                    let in_v = vld1q_f32(
                                        input.as_ptr().add(row_offset + ix),
                                    );
                                    let w_v = vld1q_f32(
                                        weights
                                            .as_ptr()
                                            .add(w_ic_offset + kiy * kw + kix),
                                    );
                                    sum = vfmaq_f32(sum, in_v, w_v);
                                } else {
                                    // Handle boundary
                                    for j in 0..4 {
                                        let idx = ix + j;
                                        if idx < w {
                                            sum = vsetq_lane_f32(
                                                vgetq_lane_f32(sum, 0)
                                                    + input[row_offset + idx]
                                                        * weights[w_ic_offset
                                                            + kiy * kw
                                                            + kix
                                                            + j],
                                                sum,
                                                0,
                                            );
                                        }
                                    }
                                }
                            }
                        } else {
                            // Scalar fallback
                            for kix in 0..kw {
                                let ix = (ow * stride + kix) as isize
                                    - padding as isize;
                                if ix >= 0 && (ix as usize) < w {
                                    let in_idx = row_offset + (ix as usize);
                                    let w_idx = w_ic_offset + kiy * kw + kix;
                                    sum = vsetq_lane_f32(
                                        vgetq_lane_f32(sum, 0)
                                            + input[in_idx] * weights[w_idx],
                                        sum,
                                        0,
                                    );
                                }
                            }
                        }
                    }
                }

                // Horizontal sum
                let result = vaddvq_f32(sum);
                output[out_row_offset + ow] = result;
            }
        }
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

    // Specialized kernels for small sizes
    if stride == 2 {
        if kh == 2 && kw == 2 {
            return unsafe { pool_2x2_neon(input, output, c, h, w) };
        }
        if kh == 3 && kw == 3 {
            return unsafe { pool_3x3_neon(input, output, c, h, w) };
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

/// 2x2 Max Pooling - matching C++ NEON intrinsics version
/// Process 4 outputs at a time, exactly like C++ NEON
#[target_feature(enable = "neon")]
unsafe fn pool_2x2_neon(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
) {
    let stride = 2;
    let h_out = (h - 2) / stride + 1;
    let w_out = (w - 2) / stride + 1;

    for ch in 0..c {
        let in_ch = input.as_ptr().add(ch * h * w);
        let out_ch = output.as_mut_ptr().add(ch * h_out * w_out);

        for oh in 0..h_out {
            let iy0 = oh * stride;
            let row0 = in_ch.add(iy0 * w);
            let row1 = in_ch.add((iy0 + 1) * w);
            let out_row = out_ch.add(oh * w_out);

            // Process 4 outputs at a time - exactly like C++
            let mut ow: usize = 0;
            while ow + 4 <= w_out {
                let ix0 = ow * stride;
                let ix1 = (ow + 1) * stride;
                let ix2 = (ow + 2) * stride;
                let ix3 = (ow + 3) * stride;

                // Manual inline to match C++ codegen
                let a0 = *row0.add(ix0);
                let a1 = *row0.add(ix0 + 1);
                let b0 = *row1.add(ix0);
                let b1 = *row1.add(ix0 + 1);
                *out_row.add(ow) = a0.max(a1).max(b0.max(b1));

                let a0 = *row0.add(ix1);
                let a1 = *row0.add(ix1 + 1);
                let b0 = *row1.add(ix1);
                let b1 = *row1.add(ix1 + 1);
                *out_row.add(ow + 1) = a0.max(a1).max(b0.max(b1));

                let a0 = *row0.add(ix2);
                let a1 = *row0.add(ix2 + 1);
                let b0 = *row1.add(ix2);
                let b1 = *row1.add(ix2 + 1);
                *out_row.add(ow + 2) = a0.max(a1).max(b0.max(b1));

                let a0 = *row0.add(ix3);
                let a1 = *row0.add(ix3 + 1);
                let b0 = *row1.add(ix3);
                let b1 = *row1.add(ix3 + 1);
                *out_row.add(ow + 3) = a0.max(a1).max(b0.max(b1));

                ow += 4;
            }

            // Handle remainder
            while ow < w_out {
                let ix = ow * stride;
                let a0 = *row0.add(ix);
                let a1 = *row0.add(ix + 1);
                let b0 = *row1.add(ix);
                let b1 = *row1.add(ix + 1);
                *out_row.add(ow) = a0.max(a1).max(b0.max(b1));
                ow += 1;
            }
        }
    }
}

/// 3x3 Max Pooling with NEON optimization
#[target_feature(enable = "neon")]
unsafe fn pool_3x3_neon(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
) {
    use std::arch::aarch64::*;

    let stride = 2;
    let h_out = (h - 3) / stride + 1;
    let w_out = (w - 3) / stride + 1;

    for ch in 0..c {
        let in_ch = input.as_ptr().add(ch * h * w);
        let out_ch = output.as_mut_ptr().add(ch * h_out * w_out);

        for oh in 0..h_out {
            let iy0 = oh * stride;
            let row0 = in_ch.add(iy0 * w);
            let row1 = in_ch.add((iy0 + 1) * w);
            let row2 = in_ch.add((iy0 + 2) * w);
            let out_row = out_ch.add(oh * w_out);

            // Process each output position
            for ow in 0..w_out {
                let ix = ow * stride;

                // Load 3 elements from each row (load 4 and ignore the last)
                let v0 = vld1q_f32(row0.add(ix)); // [ix, ix+1, ix+2, ix+3]
                let v1 = vld1q_f32(row1.add(ix));
                let v2 = vld1q_f32(row2.add(ix));

                // Max across rows (element-wise)
                let max01 = vmaxq_f32(v0, v1);
                let max012 = vmaxq_f32(max01, v2);

                // Reduce first 3 elements to single max
                // vpmaxq_f32 does pairwise max: [a,b,c,d] -> [max(a,b), max(c,d), max(a,b), max(c,d)]
                let pairwise = vpmaxq_f32(max012, max012);
                // Now pairwise = [max(el0,el1), max(el2,el3), ...]
                // We need max of first 3 elements
                let final_max = f32::max(
                    vgetq_lane_f32(pairwise, 0),
                    vgetq_lane_f32(max012, 2),
                );

                *out_row.add(ow) = final_max;
            }
        }
    }
}

/// 2D Max Pooling with NEON optimization
/// Optimized for common kernel sizes (2x2, 3x3)
///
/// DEPRECATED: This function has correctness issues and is slower than the
/// auto-vectorized scalar version. Use `pool_2d_max_f32` instead, which
/// achieves equivalent performance to C++ through compiler auto-vectorization.
#[deprecated(
    note = "Use pool_2d_max_f32 instead - auto-vectorization achieves same performance"
)]
#[target_feature(enable = "neon")]
pub unsafe fn pool_2d_max_f32_neon(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    use std::arch::aarch64::*;

    let h_out = (h - kh) / stride + 1;
    let w_out = (w - kw) / stride + 1;

    for ch in 0..c {
        let in_ch = input.as_ptr().add(ch * h * w);
        let out_ch = output.as_mut_ptr().add(ch * h_out * w_out);

        // Process output rows
        for oh in 0..h_out {
            let iy0 = oh * stride;
            let out_row = out_ch.add(oh * w_out);

            // Process output columns in batches of 4
            let mut ow = 0;

            // Handle 4 output pixels at a time
            while ow + 4 <= w_out {
                // Compute 4 max values in parallel
                let mut max_vec0 = vdupq_n_f32(f32::NEG_INFINITY);
                let mut max_vec1 = vdupq_n_f32(f32::NEG_INFINITY);
                let mut max_vec2 = vdupq_n_f32(f32::NEG_INFINITY);
                let mut max_vec3 = vdupq_n_f32(f32::NEG_INFINITY);

                let ix0 = ow * stride;
                let ix1 = (ow + 1) * stride;
                let ix2 = (ow + 2) * stride;
                let ix3 = (ow + 3) * stride;

                for kiy in 0..kh {
                    let iy = iy0 + kiy;
                    let row_ptr = in_ch.add(iy * w);

                    // Load 4x4 block for 4 output pixels
                    // For each output pixel, we need kw values starting from ix
                    // Since kernel is small (kw=2 or 3), we can load a wider block
                    if kw == 3 {
                        // Load up to 6 values: max needed is ix3+kw-1 = ix3+2
                        let v = vld1q_f32(row_ptr.add(ix0));
                        max_vec0 = vmaxq_f32(max_vec0, v);
                        max_vec0 = vmaxq_f32(
                            max_vec0,
                            vld1q_f32(row_ptr.add(ix0 + 2)),
                        );

                        let v = vld1q_f32(row_ptr.add(ix1));
                        max_vec1 = vmaxq_f32(max_vec1, v);
                        max_vec1 = vmaxq_f32(
                            max_vec1,
                            vld1q_f32(row_ptr.add(ix1 + 2)),
                        );

                        let v = vld1q_f32(row_ptr.add(ix2));
                        max_vec2 = vmaxq_f32(max_vec2, v);
                        max_vec2 = vmaxq_f32(
                            max_vec2,
                            vld1q_f32(row_ptr.add(ix2 + 2)),
                        );

                        let v = vld1q_f32(row_ptr.add(ix3));
                        max_vec3 = vmaxq_f32(max_vec3, v);
                        max_vec3 = vmaxq_f32(
                            max_vec3,
                            vld1q_f32(row_ptr.add(ix3 + 2)),
                        );
                    } else {
                        // Generic fallback for other kw
                        for kix in 0..kw {
                            max_vec0 = vsetq_lane_f32(
                                (*row_ptr.add(ix0 + kix))
                                    .max(vgetq_lane_f32(max_vec0, 0)),
                                max_vec0,
                                0,
                            );
                        }
                    }
                }

                // Reduce to scalar max
                let m0 = vmaxvq_f32(max_vec0);
                let m1 = vmaxvq_f32(max_vec1);
                let m2 = vmaxvq_f32(max_vec2);
                let m3 = vmaxvq_f32(max_vec3);

                *out_row.add(ow) = m0;
                *out_row.add(ow + 1) = m1;
                *out_row.add(ow + 2) = m2;
                *out_row.add(ow + 3) = m3;

                ow += 4;
            }

            // Tail: scalar
            while ow < w_out {
                let ix = ow * stride;
                let mut max_val = f32::NEG_INFINITY;

                for kiy in 0..kh {
                    let iy = iy0 + kiy;
                    let row_ptr = in_ch.add(iy * w);
                    for kix in 0..kw {
                        let val = *row_ptr.add(ix + kix);
                        max_val = max_val.max(val);
                    }
                }

                *out_row.add(ow) = max_val;
                ow += 1;
            }
        }
    }
}

/// 2D Average Pooling with NEON optimization
#[target_feature(enable = "neon")]
pub unsafe fn pool_2d_avg_f32_neon(
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

                    // Use NEON for processing 4 elements at a time
                    if kw >= 4 {
                        let mut sum_vec = vdupq_n_f32(0.0);
                        for kix in (0..kw).step_by(4) {
                            let v =
                                vld1q_f32(input.as_ptr().add(row_offset + kix));
                            sum_vec = vaddq_f32(sum_vec, v);
                        }
                        sum += vaddvq_f32(sum_vec);
                    } else {
                        for kix in 0..kw {
                            sum += input[row_offset + kix];
                        }
                    }
                }

                output[out_offset + oh * w_out + ow] = sum * scale;
            }
        }
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
