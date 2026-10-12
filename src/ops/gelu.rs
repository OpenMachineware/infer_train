//! GELU activation functions

#[cfg(target_arch = "aarch64")]
use super::arm::neon;

const GELU_COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_QUICK_COEF: f32 = 1.702;

/// GELU activation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x^2)))
///
/// This is the standard GELU used in most transformer models.
pub fn gelu_f32(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        let inner = 1.0 + GELU_COEF_A * x[i] * x[i];
        let tanh_arg = SQRT_2_OVER_PI * x[i] * inner;
        dst[i] = 0.5 * x[i] * (1.0 + tanh_arg.tanh());
    }
}

/// GELU with dispatch (NEON on aarch64, scalar otherwise)
pub fn gelu_f32_dispatch(x: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::gelu_f32_neon(x, dst) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gelu_f32(x, dst)
    }
}

/// Quick GELU: gelu_quick(x) = x * sigmoid(1.702 * x)
///
/// A faster approximation used by some models (e.g., GPT-2).
pub fn gelu_quick_f32(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        dst[i] = x[i] / (1.0 + (-GELU_QUICK_COEF * x[i]).exp());
    }
}

/// Quick GELU with dispatch (NEON on aarch64, scalar otherwise)
pub fn gelu_quick_f32_dispatch(x: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::gelu_quick_f32_neon(x, dst) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gelu_quick_f32(x, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu() {
        let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; x.len()];

        gelu_f32(&x, &mut dst);

        // Verify values are finite and in reasonable range
        for i in 0..x.len() {
            assert!(
                dst[i].is_finite(),
                "gelu({}) = {} is not finite",
                x[i],
                dst[i]
            );
            // GELU should be close to 0 near 0, and grow with x
            if x[i].abs() > 0.1 {
                assert!(
                    dst[i].abs() < x[i].abs() * 2.0,
                    "gelu({}) = {} too large",
                    x[i],
                    dst[i]
                );
            }
        }
    }

    #[test]
    fn test_gelu_quick() {
        let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; x.len()];

        gelu_quick_f32(&x, &mut dst);

        // Verify values are finite and in reasonable range
        for i in 0..x.len() {
            assert!(
                dst[i].is_finite(),
                "gelu_quick({}) = {} is not finite",
                x[i],
                dst[i]
            );
        }
    }
}
