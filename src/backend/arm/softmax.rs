// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Softmax operator

pub fn softmax_f32(x: &[f32], output: &mut [f32], scale: f32) {
    let n = x.len();
    assert_eq!(n, output.len());

    if n == 0 {
        return;
    }

    let mut max = x[0];
    for &v in &x[1..] {
        if v > max {
            max = v;
        }
    }

    let mut sum = 0.0f32;
    for i in 0..n {
        let val = ((x[i] * scale) - max).exp();
        output[i] = val;
        sum += val;
    }

    let inv_sum = 1.0 / sum;
    for i in 0..n {
        output[i] *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];

        softmax_f32(&x, &mut output, 1.0);

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);

        assert!((output[0] - 0.090).abs() < 0.01);
        assert!((output[1] - 0.245).abs() < 0.01);
        assert!((output[2] - 0.665).abs() < 0.01);
    }
}
