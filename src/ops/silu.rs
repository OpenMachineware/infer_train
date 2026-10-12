// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// SiLU (Swish) activation implementation

#[cfg(target_arch = "aarch64")]
use super::arm::neon;

/// Scalar SiLU: silu(x) = x / (1 + exp(-x))
pub fn silu_f32_scalar(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        dst[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

/// Auto-dispatched SiLU
pub fn silu_f32(x: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::silu_f32_neon(x, dst);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        silu_f32_scalar(x, dst);
    }
}

/// Scalar SiLU backward
pub fn silu_backward_f32_scalar(x: &[f32], dy: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        let sigmoid = 1.0 / (1.0 + (-x[i]).exp());
        let deriv = sigmoid * (1.0 + x[i] * (1.0 - sigmoid));
        dst[i] = dy[i] * deriv;
    }
}

/// Auto-dispatched SiLU backward
pub fn silu_backward_f32(x: &[f32], dy: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::silu_backward_f32_neon(x, dy, dst);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        silu_backward_f32_scalar(x, dy, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; x.len()];

        silu_f32(&x, &mut dst);

        for i in 0..x.len() {
            assert!(
                dst[i].is_finite(),
                "silu({}) = {} is not finite",
                x[i],
                dst[i]
            );
        }
    }
}
