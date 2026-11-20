// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Activation functions

/// SiLU (Swish) activation: silu(x) = x / (1 + exp(-x))
#[cfg(target_arch = "aarch64")]
pub fn silu_f32(x: &[f32], output: &mut [f32]) {
    let n = x.len();
    assert_eq!(n, output.len());

    // Use scalar implementation - compiler auto-vectorizes well
    for i in 0..n {
        output[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

/// In-place SiLU activation
#[cfg(target_arch = "aarch64")]
pub fn silu_f32_inplace(x: &mut [f32]) {
    for i in 0..x.len() {
        x[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

/// Fused SwiGLU: output = silu(gate) * up
#[cfg(target_arch = "aarch64")]
pub fn swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), output.len());

    let n = gate.len();
    for i in 0..n {
        let silu = gate[i] / (1.0 + (-gate[i]).exp());
        output[i] = silu * up[i];
    }
}

// Fallback scalar implementations
#[cfg(not(target_arch = "aarch64"))]
pub fn silu_f32(x: &[f32], output: &mut [f32]) {
    let n = x.len();
    assert_eq!(n, output.len());

    for i in 0..n {
        output[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn silu_f32_inplace(x: &mut [f32]) {
    for i in 0..x.len() {
        x[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), output.len());

    let n = gate.len();
    for i in 0..n {
        let silu = gate[i] / (1.0 + (-gate[i]).exp());
        output[i] = silu * up[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        // Use multiple of 4 elements to test NEON path
        let x = vec![-2.0f32, -1.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];

        silu_f32(&x, &mut output);

        // silu(0) = 0
        assert!((output[2]).abs() < 1e-5);
        // silu(1) = 1/(1+exp(-1)) ≈ 0.731
        assert!((output[3] - 0.731).abs() < 0.01);
        // silu(-1) ≈ -0.269
        assert!((output[1] - (-0.269)).abs() < 0.01);
        // silu(-2) ≈ -0.238
        assert!((output[0] - (-0.238)).abs() < 0.02);
    }

    #[test]
    fn test_swiglu() {
        let gate = vec![-1.0f32, 0.0, 1.0, 2.0];
        let up = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];

        swiglu_f32(&gate, &up, &mut output);

        // silu(0) = 0, so output[1] = 0 * 2 = 0
        assert!((output[1]).abs() < 1e-5);
        // silu(1) ≈ 0.731, so output[2] ≈ 0.731 * 3 ≈ 2.19
        assert!((output[2] - 2.19).abs() < 0.1);
    }
}
