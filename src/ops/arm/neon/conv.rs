// NEON SIMD implementations for CNN operators

use std::arch::aarch64::*;

/// 2D Convolution with NEON optimization
#[target_feature(enable = "neon")]
pub unsafe fn conv_2d_f32(
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

/// 2x2 Max Pooling with NEON optimization
#[target_feature(enable = "neon")]
pub unsafe fn pool_2x2(
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

            // Process 4 outputs at a time
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
pub unsafe fn pool_3x3(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
) {
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
                let v0 = vld1q_f32(row0.add(ix));
                let v1 = vld1q_f32(row1.add(ix));
                let v2 = vld1q_f32(row2.add(ix));

                // Max across rows (element-wise)
                let max01 = vmaxq_f32(v0, v1);
                let max012 = vmaxq_f32(max01, v2);

                // Reduce first 3 elements to single max
                let pairwise = vpmaxq_f32(max012, max012);
                let final_max = f32::max(
                    vgetq_lane_f32(pairwise, 0),
                    vgetq_lane_f32(max012, 2),
                );

                *out_row.add(ow) = final_max;
            }
        }
    }
}

/// 2D Average Pooling with NEON optimization
#[target_feature(enable = "neon")]
pub unsafe fn pool_2d_avg_f32(
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
