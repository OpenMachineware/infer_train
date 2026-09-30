//! CNN operators: CONV_2D, CONV_2D_DW, POOL_2D, IM2COL

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
    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;

    for oc in 0..c_out {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut sum = bias.map(|b| b[oc]).unwrap_or(0.0);

                for ic in 0..c_in {
                    for kiy in 0..kh {
                        for kix in 0..kw {
                            let iy = oh * stride + kiy;
                            let ix = ow * stride + kix;

                            // Padding check
                            let iy = iy as isize - padding as isize;
                            let ix = ix as isize - padding as isize;

                            if iy >= 0
                                && ix >= 0
                                && (iy as usize) < h
                                && (ix as usize) < w
                            {
                                let in_idx = ic * h * w
                                    + (iy as usize) * w
                                    + (ix as usize);
                                let w_idx = oc * c_in * kh * kw
                                    + ic * kh * kw
                                    + kiy * kw
                                    + kix;
                                sum += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                }

                let out_idx = oc * h_out * w_out + oh * w_out + ow;
                output[out_idx] = sum;
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

/// 2D Pooling (max or average)
#[derive(Clone, Copy)]
pub enum PoolType {
    Max,
    Avg,
}

pub fn pool_2d_f32(
    input: &[f32],
    output: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pool_type: PoolType,
) {
    let h_out = (h - kh) / stride + 1;
    let w_out = (w - kw) / stride + 1;

    for ch in 0..c {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut vals: Vec<f32> = Vec::with_capacity(kh * kw);

                for kiy in 0..kh {
                    for kix in 0..kw {
                        let iy = oh * stride + kiy;
                        let ix = ow * stride + kix;
                        let idx = ch * h * w + iy * w + ix;
                        vals.push(input[idx]);
                    }
                }

                let result = match pool_type {
                    PoolType::Max => {
                        vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    }
                    PoolType::Avg => {
                        vals.iter().sum::<f32>() / vals.len() as f32
                    }
                };

                let out_idx = ch * h_out * w_out + oh * w_out + ow;
                output[out_idx] = result;
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

        pool_2d_f32(&input, &mut output, 1, 4, 4, 2, 2, 2, PoolType::Max);

        assert_eq!(output, [6.0, 8.0, 14.0, 16.0]);
    }
}
