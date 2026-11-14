// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Activation functions

pub fn silu_f32(x: &[f32], output: &mut [f32]) {
    let n = x.len();
    assert_eq!(n, output.len());

    for i in 0..n {
        output[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        let x = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 5];

        silu_f32(&x, &mut output);

        assert!((output[2]).abs() < 1e-6);
        assert!((output[3] - 0.731).abs() < 0.01);
        assert!((output[4] - 1.761).abs() < 0.01);
        assert!((output[1] - (-0.269)).abs() < 0.01);
    }
}
