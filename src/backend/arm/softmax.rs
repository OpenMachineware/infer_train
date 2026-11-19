// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Softmax operator - NEON optimized

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Vectorized exp for NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vexpq_f32(x: float32x4_t) -> float32x4_t {
    let r = vdupq_n_f32(12582912.0);
    let c1 = vdupq_n_f32(1.4426950408889634);
    let c2 = vdupq_n_f32(0.6931381225585938);
    let c3 = vdupq_n_f32(1.908214e-10);

    let z = vfmaq_f32(r, x, c1);
    let n = vsubq_f32(z, r);
    let b = vfmsq_f32(vfmsq_f32(x, n, c2), n, c3);

    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(
        e,
        vreinterpretq_u32_f32(vdupq_n_f32(1.0f32)),
    ));

    let c = vcagtq_f32(n, vdupq_n_f32(126.0f32));

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

    if vpaddd_u64(vreinterpretq_u64_u32(c)) == 0 {
        return vfmaq_f32(k, j, k);
    }

    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000u32));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000u32)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));

    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0f32)),
        vmulq_f32(s1, s1),
        vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
    )
}

/// NEON SIMD softmax with optional scale
#[cfg(target_arch = "aarch64")]
pub fn softmax_f32(x: &[f32], output: &mut [f32], scale: f32) {
    let n = x.len();
    assert_eq!(n, output.len());

    if n == 0 {
        return;
    }

    unsafe {
        // Find max using NEON
        let mut max = f32::NEG_INFINITY;
        let mut i = 0;

        for chunk in x.chunks_exact(4) {
            let v = vld1q_f32(chunk.as_ptr());
            let chunk_max = vmaxvq_f32(v);
            if chunk_max > max {
                max = chunk_max;
            }
            i += 4;
        }

        for &xi in &x[i..] {
            if xi > max {
                max = xi;
            }
        }

        // Scale max if needed
        if scale != 1.0 {
            max *= scale;
        }

        // Compute exp(x * scale - max) and sum
        let mut sum = 0.0f32;
        i = 0;

        let vscale = vdupq_n_f32(scale);
        let vmax = vdupq_n_f32(max);

        for (x_chunk, y_chunk) in
            x.chunks_exact(4).zip(output.chunks_exact_mut(4))
        {
            let vx = vld1q_f32(x_chunk.as_ptr());
            let vscaled = vmulq_f32(vx, vscale);
            let vsub = vsubq_f32(vscaled, vmax);
            let vexp = vexpq_f32(vsub);
            vst1q_f32(y_chunk.as_mut_ptr(), vexp);
            sum += vaddvq_f32(vexp);
            i += 4;
        }

        for j in i..n {
            let val = ((x[j] * scale) - max).exp();
            output[j] = val;
            sum += val;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        i = 0;

        for y_chunk in output.chunks_exact_mut(4) {
            let vy = vld1q_f32(y_chunk.as_ptr());
            let vinv = vdupq_n_f32(inv_sum);
            let vscaled = vmulq_f32(vy, vinv);
            vst1q_f32(y_chunk.as_mut_ptr(), vscaled);
            i += 4;
        }

        for yi in output[i..].iter_mut() {
            *yi *= inv_sum;
        }
    }
}

// Fallback scalar implementation
#[cfg(not(target_arch = "aarch64"))]
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

    if scale != 1.0 {
        max *= scale;
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

    #[test]
    fn test_softmax_with_scale() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];

        softmax_f32(&x, &mut output, 0.5);

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
