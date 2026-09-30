//! Other operators: NORM, GROUP_NORM, CONCAT, REPEAT, CLAMP

/// Layer Normalization (standard, not RMS)
pub fn layer_norm_f32(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    y: &mut [f32],
    n: usize,
) {
    let eps = 1e-5f32;

    for i in 0..n {
        // Compute mean
        let mean: f32 = x.iter().sum::<f32>() / n as f32;

        // Compute variance
        let var: f32 =
            x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / n as f32;

        // Normalize
        let std = (var + eps).sqrt();
        y[i] = (x[i] - mean) / std * weight[i] + bias[i];
    }
}

/// Group Normalization
/// Divides channels into groups and normalizes within each group
pub fn group_norm_f32(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    y: &mut [f32],
    n_groups: usize,
    n_channels: usize,
    hw: usize, // height * width
) {
    let group_size = n_channels / n_groups;
    let eps = 1e-5f32;

    for g in 0..n_groups {
        // Compute mean and variance over the group
        let group_offset = g * group_size * hw;
        let group_len = group_size * hw;

        let group_slice = &x[group_offset..group_offset + group_len];
        let mean: f32 = group_slice.iter().sum::<f32>() / group_len as f32;
        let var: f32 =
            group_slice.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>()
                / group_len as f32;

        let std = (var + eps).sqrt();

        // Normalize each channel in the group
        for c in 0..group_size {
            let ch = g * group_size + c;
            for s in 0..hw {
                let idx = ch * hw + s;
                y[idx] = (x[idx] - mean) / std * weight[ch] + bias[ch];
            }
        }
    }
}

/// Concatenate tensors along a dimension
pub fn concat_f32(
    inputs: &[&[f32]],
    output: &mut [f32],
    n_tensors: usize,
    len: usize,
) {
    let mut offset = 0;
    for i in 0..n_tensors {
        output[offset..offset + len].copy_from_slice(inputs[i]);
        offset += len;
    }
}

/// Repeat tensor along a dimension
pub fn repeat_f32(
    input: &[f32],
    output: &mut [f32],
    n_repeat: usize,
    len: usize,
) {
    for i in 0..n_repeat {
        output[i * len..(i + 1) * len].copy_from_slice(input);
    }
}

/// Clamp values to a range
pub fn clamp_f32(input: &[f32], output: &mut [f32], min: f32, max: f32) {
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = if i < min {
            min
        } else if i > max {
            max
        } else {
            i
        };
    }
}

/// Fill tensor with a value
pub fn fill_f32(output: &mut [f32], value: f32) {
    for o in output.iter_mut() {
        *o = value;
    }
}

/// Pad tensor with zeros
pub fn pad_f32(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let h_out = h + 2 * pad_h;
    let w_out = w + 2 * pad_w;

    // Fill with zeros
    output.fill(0.0);

    // Copy input to center
    for i in 0..h {
        let src_row = i * w;
        let dst_row = (i + pad_h) * w_out + pad_w;
        output[dst_row..dst_row + w]
            .copy_from_slice(&input[src_row..src_row + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let b = [0.0, 0.0, 0.0, 0.0];
        let mut y = [0.0f32; 4];

        layer_norm_f32(&x, &w, &b, &mut y, 4);

        // Mean = 2.5, variance = 1.25
        // Normalized values should have zero mean
        let sum: f32 = y.iter().sum();
        assert!(sum.abs() < 1e-4);
    }

    #[test]
    fn test_clamp() {
        let input = [0.0, 0.5, 1.0, 1.5, 2.0];
        let mut output = [0.0f32; 5];

        clamp_f32(&input, &mut output, 0.25, 1.25);

        assert_eq!(output, [0.25, 0.5, 1.0, 1.25, 1.25]);
    }

    #[test]
    fn test_repeat() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 6];

        repeat_f32(&input, &mut output, 2, 3);

        assert_eq!(output, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
}
