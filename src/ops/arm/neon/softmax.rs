// NEON SIMD softmax implementation

use std::arch::aarch64::*;

/// NEON SIMD softmax
/// Uses the same exp approximation as llama.cpp's ggml_v_expf
#[target_feature(enable = "neon")]
pub unsafe fn softmax_f32(x: &[f32], y: &mut [f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    if n == 0 {
        return 0.0;
    }

    // Find max using NEON
    let mut max = f32::NEG_INFINITY;
    let mut i = 0;

    // Process 4 elements at a time
    for chunk in x.chunks_exact(4) {
        let v = vld1q_f32(chunk.as_ptr());
        let chunk_max = vmaxvq_f32(v);
        if chunk_max > max {
            max = chunk_max;
        }
        i += 4;
    }

    // Handle remainder
    for &xi in &x[i..] {
        if xi > max {
            max = xi;
        }
    }

    // Compute exp(x - max) and sum using NEON
    let mut sum = 0.0f32;
    i = 0;

    for (x_chunk, y_chunk) in x.chunks_exact(4).zip(y.chunks_exact_mut(4)) {
        let vx = vld1q_f32(x_chunk.as_ptr());
        let vmax = vdupq_n_f32(max);
        let vsub = vsubq_f32(vx, vmax);
        let vexp = exp(vsub);
        vst1q_f32(y_chunk.as_mut_ptr(), vexp);
        sum += vaddvq_f32(vexp);
        i += 4;
    }

    // Handle remainder with scalar
    for j in i..n {
        let val = (x[j] - max).exp();
        y[j] = val;
        sum += val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    i = 0;

    for y_chunk in y.chunks_exact_mut(4) {
        let vy = vld1q_f32(y_chunk.as_ptr());
        let vinv = vdupq_n_f32(inv_sum);
        let vscaled = vmulq_f32(vy, vinv);
        vst1q_f32(y_chunk.as_mut_ptr(), vscaled);
        i += 4;
    }

    for yi in y[i..].iter_mut() {
        *yi *= inv_sum;
    }

    sum
}

/// NEON exp approximation from llama.cpp's ggml_v_expf
/// Uses Payne-Hanek polynomial approximation
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn exp(x: float32x4_t) -> float32x4_t {
    // Constants for exp approximation (converted from hex floats)
    let r = vdupq_n_f32(12582912.0); // 0x1.8p23f = 2^23 * 1.5
    let c1 = vdupq_n_f32(1.4426950408889634); // 1/ln(2)
    let c2 = vdupq_n_f32(0.6931381225585938); // ln(2) high part
    let c3 = vdupq_n_f32(1.908214e-10); // ln(2) low part correction

    // Compute n = round(x * 1/ln(2))
    let z = vfmaq_f32(r, x, c1);
    let n = vsubq_f32(z, r);

    // Compute reduced argument
    let b = vfmsq_f32(vfmsq_f32(x, n, c2), n, c3);

    // Compute 2^n
    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(
        e,
        vreinterpretq_u32_f32(vdupq_n_f32(1.0f32)),
    ));

    // Check for overflow/underflow
    let c = vcagtq_f32(n, vdupq_n_f32(126.0f32));

    // Polynomial approximation: exp(b) ≈ 1 + b + b^2/2! + b^3/3! + ...
    // Using Horner's method with coefficients from llama.cpp
    let u = vmulq_f32(b, b);
    let j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0.9999990463256836), b),
        vfmaq_f32(
            vfmaq_f32(
                vdupq_n_f32(0.4999985098838806),
                vdupq_n_f32(0.1666666169166565),
                b,
            ),
            vfmaq_f32(
                vdupq_n_f32(0.04166661170125008),
                vdupq_n_f32(0.008333338060230017),
                b,
            ),
            u,
        ),
        u,
    );

    // Check if any lane needs special handling
    if vpaddd_u64(vreinterpretq_u64_u32(c)) == 0 {
        // Normal case: return k * (1 + j)
        return vfmaq_f32(k, j, k);
    }

    // Handle overflow/underflow
    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000u32));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000u32)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));

    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0f32)),
        vmulq_f32(s1, s1), // overflow -> inf
        vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
    )
}
