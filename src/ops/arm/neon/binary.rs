// NEON SIMD implementations for binary operations (for non-macOS ARM64)

use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn add_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let a = vld1q_f32(src0.as_ptr().add(offset));
        let b = vld1q_f32(src1.as_ptr().add(offset));
        let result = vaddq_f32(a, b);
        vst1q_f32(dst.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        dst[i] = src0[i] + src1[i];
    }
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sub_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let a = vld1q_f32(src0.as_ptr().add(offset));
        let b = vld1q_f32(src1.as_ptr().add(offset));
        let result = vsubq_f32(a, b);
        vst1q_f32(dst.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        dst[i] = src0[i] - src1[i];
    }
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn mul_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let a = vld1q_f32(src0.as_ptr().add(offset));
        let b = vld1q_f32(src1.as_ptr().add(offset));
        let result = vmulq_f32(a, b);
        vst1q_f32(dst.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        dst[i] = src0[i] * src1[i];
    }
}

#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn div_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let a = vld1q_f32(src0.as_ptr().add(offset));
        let b = vld1q_f32(src1.as_ptr().add(offset));
        let result = vdivq_f32(a, b);
        vst1q_f32(dst.as_mut_ptr().add(offset), result);
    }

    for i in (chunks * 4)..n {
        dst[i] = src0[i] / src1[i];
    }
}
