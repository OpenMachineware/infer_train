use std::arch::aarch64::*;

// Constants for exp approximation (from llama.cpp)
const EXP_R: f32 = 12582912.0;  // 0x1.8p23f
const EXP_C1: f32 = 1.442695;   // 0x1.715476p+0f
const EXP_C2: f32 = 0.4434359;  // 0x1.62e4p-1f - 0x1.7f7d1cp-20f
const EXP_C3: f32 = 1.0;
const EXP_POLY_0: f32 = 0.9999998;  // 0x1.ffffecp-1f
const EXP_POLY_1: f32 = 0.4999905;  // 0x1.fffdb6p-2f
const EXP_POLY_2: f32 = 0.1666656;  // 0x1.555e66p-3f
const EXP_POLY_3: f32 = 0.0418799;  // 0x1.573e2ep-5f
const EXP_POLY_4: f32 = 0.0083356;  // 0x1.0e4020p-7f

/// Vectorized exp function for NEON
/// Adapted from ARM optimized routine
/// Maximum error: 1.45358 plus 0.5 ulps
/// Numbers above 88.38 flush to infinity
/// Numbers below -103.97 flush to zero
#[target_feature(enable = "neon")]
pub unsafe fn vexpq_f32(x: float32x4_t) -> float32x4_t {
    let r = vdupq_n_f32(EXP_R);
    let z = vfmaq_f32(r, x, vdupq_n_f32(EXP_C1));
    let n = vsubq_f32(z, r);
    let b = vfmsq_f32(
        vfmsq_f32(x, n, vdupq_n_f32(0.889301)),
        n,
        vdupq_n_f32(1.69142e-6),
    );
    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(
        e,
        vreinterpretq_u32_f32(vdupq_n_f32(1.0)),
    ));
    let c = vcagtq_f32(n, vdupq_n_f32(126.0));
    let u = vmulq_f32(b, b);
    let j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(EXP_POLY_0), b),
        vfmaq_f32(
            vfmaq_f32(vdupq_n_f32(EXP_POLY_1), vdupq_n_f32(EXP_POLY_2), b),
            vfmaq_f32(vdupq_n_f32(EXP_POLY_3), vdupq_n_f32(EXP_POLY_4), b),
            u,
        ),
        u,
    );

    if vpaddd_u64(vreinterpretq_u64_u32(c)) == 0 {
        return vfmaq_f32(k, j, k);
    }

    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));

    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0)),
        vmulq_f32(s1, s1),
        vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
    )
}

/// SiLU (Swish) activation: silu(x) = x / (1 + exp(-x))
///
/// # Arguments
/// * `x` - Input tensor
/// * `dst` - Output tensor
#[target_feature(enable = "neon")]
pub unsafe fn silu_f32(x: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    // Process 4 elements at a time
    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * 4));
        let neg_x = vsubq_f32(zero, vx);
        let exp_neg_x = vexpq_f32(neg_x);
        let one_plus_exp = vaddq_f32(one, exp_neg_x);
        let result = vdivq_f32(vx, one_plus_exp);
        vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        dst[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

/// SiLU backward: d(silu)/dx = silu(x) + x * sigmoid(x) * (1 - sigmoid(x))
/// where silu(x) = x * sigmoid(x), sigmoid(x) = 1/(1+exp(-x))
///
/// Simplified: d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#[target_feature(enable = "neon")]
pub unsafe fn silu_backward_f32(x: &[f32], dy: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * 4));
        let vdy = vld1q_f32(dy.as_ptr().add(i * 4));

        let neg_x = vsubq_f32(zero, vx);
        let exp_neg_x = vexpq_f32(neg_x);
        let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_neg_x));
        let one_minus_sigmoid = vsubq_f32(one, sigmoid);
        let x_mul = vmulq_f32(vx, one_minus_sigmoid);
        let deriv = vmulq_f32(sigmoid, vaddq_f32(one, x_mul));
        let result = vmulq_f32(vdy, deriv);

        vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
    }

    for i in (chunks * 4)..n {
        let sigmoid = 1.0 / (1.0 + (-x[i]).exp());
        let deriv = sigmoid * (1.0 + x[i] * (1.0 - sigmoid));
        dst[i] = dy[i] * deriv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; x.len()];

        unsafe {
            silu_f32(&x, &mut dst);
        }

        // Verify against reference
        for i in 0..x.len() {
            let expected = x[i] / (1.0 + (-x[i]).exp());
            assert!(
                (dst[i] - expected).abs() < 1e-4,
                "silu({}) = {}, expected {}",
                x[i],
                dst[i],
                expected
            );
        }
    }
}
