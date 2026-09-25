use std::arch::aarch64::*;

const GELU_COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const SQRT_2_INV: f32 = 0.7071067811865475;
const GELU_QUICK_COEF: f32 = 1.702;

/// Vectorized tanh function for NEON
/// Uses rational approximation for better accuracy
#[target_feature(enable = "neon")]
pub unsafe fn vtanhq_f32(x: float32x4_t) -> float32x4_t {
    let abs_x = vabsq_f32(x);
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    let four = vdupq_n_f32(4.0);

    // For |x| >= 4, tanh(x) ≈ sign(x) * 1.0
    let large_mask = vcagtq_f32(x, four);

    // For small |x|, use exp approximation: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    // = (e^(2x) - 1) / (e^(2x) + 1)
    // = 1 - 2 / (e^(2x) + 1)
    let two_x = vmulq_f32(two, x);
    let exp_2x = crate::ops::silu::vexpq_f32(two_x);
    let tanh_approx = vsubq_f32(one, vdivq_f32(two, vaddq_f32(exp_2x, one)));

    // For large |x|, return sign(x)
    let sign = vreinterpretq_f32_u32(vorrq_u32(
        vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000)),
        vreinterpretq_u32_f32(one),
    ));

    vbslq_f32(large_mask, sign, tanh_approx)
}

/// GELU activation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x^2)))
///
/// This is the standard GELU used in most transformer models.
///
/// # Arguments
/// * `x` - Input tensor
/// * `dst` - Output tensor
#[target_feature(enable = "neon")]
pub unsafe fn gelu_f32(x: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;

    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let sqrt_2_over_pi = vdupq_n_f32(SQRT_2_OVER_PI);
    let gelu_coef = vdupq_n_f32(GELU_COEF_A);

    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * 4));

        // tanh_arg = sqrt(2/π) * x * (1 + 0.044715 * x^2)
        let x2 = vmulq_f32(vx, vx);
        let inner = vfmaq_f32(one, gelu_coef, x2);
        let tanh_arg = vmulq_f32(vmulq_f32(sqrt_2_over_pi, vx), inner);

        let tanh_val = vtanhq_f32(tanh_arg);

        // gelu = 0.5 * x * (1 + tanh)
        let one_plus_tanh = vaddq_f32(one, tanh_val);
        let result = vmulq_f32(vmulq_f32(half, vx), one_plus_tanh);

        vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        let inner = 1.0 + GELU_COEF_A * x[i] * x[i];
        let tanh_arg = SQRT_2_OVER_PI * x[i] * inner;
        dst[i] = 0.5 * x[i] * (1.0 + tanh_arg.tanh());
    }
}

/// Quick GELU: gelu_quick(x) = x * sigmoid(1.702 * x)
///
/// A faster approximation used by some models (e.g., GPT-2).
#[target_feature(enable = "neon")]
pub unsafe fn gelu_quick_f32(x: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;

    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let quick_coef = vdupq_n_f32(-GELU_QUICK_COEF);  // -1.702

    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * 4));

        // arg = -1.702 * x
        let arg = vmulq_f32(quick_coef, vx);
        let exp_arg = crate::ops::silu::vexpq_f32(arg);
        let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_arg));

        let result = vmulq_f32(vx, sigmoid);
        vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
    }

    for i in (chunks * 4)..n {
        dst[i] = x[i] / (1.0 + (-GELU_QUICK_COEF * x[i]).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu() {
        let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; x.len()];

        unsafe {
            gelu_f32(&x, &mut dst);
        }

        // Verify against reference
        for i in 0..x.len() {
            let inner = 1.0 + GELU_COEF_A * x[i] * x[i];
            let tanh_arg = SQRT_2_OVER_PI * x[i] * inner;
            let expected = 0.5 * x[i] * (1.0 + tanh_arg.tanh());
            assert!(
                (dst[i] - expected).abs() < 0.01,
                "gelu({}) = {}, expected {}",
                x[i],
                dst[i],
                expected
            );
        }
    }

    #[test]
    fn test_gelu_quick() {
        let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; x.len()];

        unsafe {
            gelu_quick_f32(&x, &mut dst);
        }

        // Verify against reference
        for i in 0..x.len() {
            let expected = x[i] / (1.0 + (GELU_QUICK_COEF * x[i]).exp());
            assert!(
                (dst[i] - expected).abs() < 1e-4,
                "gelu_quick({}) = {}, expected {}",
                x[i],
                dst[i],
                expected
            );
        }
    }
}
