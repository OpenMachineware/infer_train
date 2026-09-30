use half::f16;
use std::arch::aarch64::*;
use std::arch::asm;

/// F16 GEMM: C = A^T * B (output in F32)
///
/// - A: [m, k] in row-major, stored as F16
/// - B: [n, k] in row-major, stored as F16
/// - C: [n, m] in row-major, stored as F32
///
/// Uses inline assembly for F16 FMLA (unstable in stable Rust).
/// Requires n >= 8 for vectorized path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hgemm(
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // For small n, fallback to scalar
    if n < 8 {
        hgemm_scalar(a, b, c, m, n, k);
        return;
    }

    // Block tiling for F16: KN=8 (8 F16 per float16x8_t)
    const RM: usize = 4;
    const RN: usize = 2;

    for i0 in (0..m).step_by(RM) {
        let i1 = (i0 + RM).min(m);

        for j0 in (0..n).step_by(RN) {
            let j1 = (j0 + RN).min(n);
            hgemm_bloc(a, b, c, m, k, i0, i1, j0, j1);
        }
    }
}

/// Inline assembly for F16 FMLA
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vfmaq_f16_asm(
    a: float16x8_t,
    b: float16x8_t,
    c: float16x8_t,
) -> float16x8_t {
    let result: float16x8_t;
    asm!(
        "fmla {0:v}.8h, {1:v}.8h, {2:v}.8h",
        inout(vreg) a => result,
        in(vreg) b,
        in(vreg) c,
        options(pure, nomem, nostack)
    );
    result
}

/// Inline assembly for F16 horizontal sum
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vaddvq_f16_asm(v: float16x8_t) -> f32 {
    // Convert to F32 and sum
    let low = vcvt_f32_f16(vget_low_f16(v));
    let high = vcvt_f32_f16(vget_high_f16(v));
    vaddvq_f32(vaddq_f32(low, high))
}

/// Create zero F16 vector using inline assembly
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vdupq_n_f16_zero() -> float16x8_t {
    // Use movi to create zero vector (interpreted as F16)
    vreinterpretq_f16_u16(vdupq_n_u16(0))
}

/// F16 GEMM block computation using inline assembly
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hgemm_bloc(
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
    ldc: usize,
    k: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let rm = i1 - i0;
    let rn = j1 - j0;

    // SIMD accumulators in F16
    let zero = vdupq_n_f16_zero();
    let mut cv: [[float16x8_t; 4]; 2] =
        [[zero, zero, zero, zero], [zero, zero, zero, zero]];

    // Main loop: process k in steps of 8
    let k8 = k - k % 8;
    for l in (0..k8).step_by(8) {
        // Load 4 rows from A (half::f16 is stored as u16)
        let av: [float16x8_t; 4] = [
            if i0 + 0 < i1 {
                vreinterpretq_f16_u16(vld1q_u16(
                    a.as_ptr().add((i0 + 0) * k + l) as *const u16,
                ))
            } else {
                zero
            },
            if i0 + 1 < i1 {
                vreinterpretq_f16_u16(vld1q_u16(
                    a.as_ptr().add((i0 + 1) * k + l) as *const u16,
                ))
            } else {
                zero
            },
            if i0 + 2 < i1 {
                vreinterpretq_f16_u16(vld1q_u16(
                    a.as_ptr().add((i0 + 2) * k + l) as *const u16,
                ))
            } else {
                zero
            },
            if i0 + 3 < i1 {
                vreinterpretq_f16_u16(vld1q_u16(
                    a.as_ptr().add((i0 + 3) * k + l) as *const u16,
                ))
            } else {
                zero
            },
        ];

        // For each column, load B and do FMLA
        for j in 0..rn {
            let bv = vreinterpretq_f16_u16(vld1q_u16(
                b.as_ptr().add((j0 + j) * k + l) as *const u16,
            ));
            cv[j][0] = vfmaq_f16_asm(cv[j][0], av[0], bv);
            cv[j][1] = vfmaq_f16_asm(cv[j][1], av[1], bv);
            cv[j][2] = vfmaq_f16_asm(cv[j][2], av[2], bv);
            cv[j][3] = vfmaq_f16_asm(cv[j][3], av[3], bv);
        }
    }

    // Convert accumulators to F32 and store
    for j in 0..rn {
        for i in 0..rm {
            // Convert F16 accumulator to F32 and sum
            let sum = vaddvq_f16_asm(cv[j][i]);

            // Handle remaining k
            let mut tail = 0.0f32;
            for l in k8..k {
                tail +=
                    a[(i0 + i) * k + l].to_f32() * b[(j0 + j) * k + l].to_f32();
            }

            c[(j0 + j) * ldc + (i0 + i)] = sum + tail;
        }
    }
}

/// Scalar fallback for small n
#[cfg(target_arch = "aarch64")]
unsafe fn hgemm_scalar(
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l].to_f32() * b[j * k + l].to_f32();
            }
            c[j * m + i] = sum;
        }
    }
}

/// F16 × F32 GEMM: A is F16, B is F32, C is F32
/// Used when B is in F32 but A is quantized to F16
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hgemm_f32b(
    a: &[f16],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    if n < 4 {
        // Scalar fallback
        for j in 0..n {
            for i in 0..m {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l].to_f32() * b[j * k + l];
                }
                c[j * m + i] = sum;
            }
        }
        return;
    }

    const RM: usize = 4;
    const RN: usize = 3;

    for i0 in (0..m).step_by(RM) {
        let i1 = (i0 + RM).min(m);

        for j0 in (0..n).step_by(RN) {
            let j1 = (j0 + RN).min(n);
            hgemm_f32b_bloc(a, b, c, m, k, i0, i1, j0, j1);
        }
    }
}

/// F16 × F32 GEMM block
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hgemm_f32b_bloc(
    a: &[f16],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    k: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let rm = i1 - i0;
    let rn = j1 - j0;

    // Accumulators in F32
    let zero = vdupq_n_f32(0.0);
    let mut cv: [[float32x4_t; 4]; 3] = [
        [zero, zero, zero, zero],
        [zero, zero, zero, zero],
        [zero, zero, zero, zero],
    ];

    let k4 = k - k % 4;
    for l in (0..k4).step_by(4) {
        // Load F16 from A and convert to F32 (half::f16 is stored as u16)
        let av: [float32x4_t; 4] = [
            if i0 + 0 < i1 {
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                    a.as_ptr().add((i0 + 0) * k + l) as *const u16,
                )))
            } else {
                zero
            },
            if i0 + 1 < i1 {
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                    a.as_ptr().add((i0 + 1) * k + l) as *const u16,
                )))
            } else {
                zero
            },
            if i0 + 2 < i1 {
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                    a.as_ptr().add((i0 + 2) * k + l) as *const u16,
                )))
            } else {
                zero
            },
            if i0 + 3 < i1 {
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(
                    a.as_ptr().add((i0 + 3) * k + l) as *const u16,
                )))
            } else {
                zero
            },
        ];

        // Load F32 from B and do FMLA
        for j in 0..rn {
            let bv = vld1q_f32(b.as_ptr().add((j0 + j) * k + l));
            cv[j][0] = vfmaq_f32(cv[j][0], av[0], bv);
            cv[j][1] = vfmaq_f32(cv[j][1], av[1], bv);
            cv[j][2] = vfmaq_f32(cv[j][2], av[2], bv);
            cv[j][3] = vfmaq_f32(cv[j][3], av[3], bv);
        }
    }

    // Horizontal sum and store
    for j in 0..rn {
        for i in 0..rm {
            let sum = vaddvq_f32(cv[j][i]);

            let mut tail = 0.0f32;
            for l in k4..k {
                tail += a[(i0 + i) * k + l].to_f32() * b[(j0 + j) * k + l];
            }

            c[(j0 + j) * ldc + (i0 + i)] = sum + tail;
        }
    }
}
