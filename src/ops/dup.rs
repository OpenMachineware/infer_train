// DUP/CPY operations: copy with type conversion
// Matches llama.cpp's ggml_compute_forward_dup

use half::f16;

/// Type conversion trait
pub trait FloatType: Copy + Clone {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl FloatType for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 { self }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { v }
}

impl FloatType for f16 {
    #[inline(always)]
    fn to_f32(self) -> f32 { self.to_f32() }
    #[inline(always)]
    fn from_f32(v: f32) -> Self { f16::from_f32(v) }
}

/// Copy same type (memcpy)
#[inline]
pub fn dup_same<T: Copy>(src: &[T], dst: &mut [T]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
}

/// Copy with type conversion: F32 -> F16
pub fn dup_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = f16::from_f32(src[i]);
    }
}

/// Copy with type conversion: F16 -> F32
pub fn dup_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = src[i].to_f32();
    }
}

/// Generic copy with type conversion
pub fn dup<S, D>(src: &[S], dst: &mut [D])
where
    S: FloatType,
    D: FloatType,
{
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = D::from_f32(src[i].to_f32());
    }
}

/// NEON optimized F32 -> F16 conversion
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dup_f32_to_f16_neon(src: &[f32], dst: &mut [f16]) {
    let n = dst.len().min(src.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let f32_vec = vld1q_f32(src.as_ptr().add(offset));
        // Convert to f16 using scalar (NEON f16 intrinsics are unstable)
        for j in 0..4 {
            dst[offset + j] = f16::from_f32(src[offset + j]);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        dst[i] = f16::from_f32(src[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dup_f16_to_f32_neon(src: &[f16], dst: &mut [f32]) {
    let n = dst.len().min(src.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        // Convert using scalar (NEON f16 intrinsics are unstable)
        for j in 0..4 {
            dst[offset + j] = src[offset + j].to_f32();
        }
    }

    for i in (chunks * 4)..n {
        dst[i] = src[i].to_f32();
    }
}

/// Public API for F32->F16 conversion
#[cfg(target_arch = "aarch64")]
pub fn convert_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    // Scalar is fast enough for now (half crate is optimized)
    dup_f32_to_f16(src, dst);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn convert_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    dup_f32_to_f16(src, dst);
}

/// Public API for F16->F32 conversion
#[cfg(target_arch = "aarch64")]
pub fn convert_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    dup_f16_to_f32(src, dst);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn convert_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    dup_f16_to_f32(src, dst);
}

// CPY is just an alias for DUP in most cases
pub use dup as cpy;
pub use dup_same as cpy_same;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dup_f32() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];

        dup_same(&src, &mut dst);

        for i in 0..16 {
            assert_eq!(dst[i], src[i]);
        }
    }

    #[test]
    fn test_dup_f32_f16() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let mut dst = vec![f16::ZERO; 16];

        dup_f32_to_f16(&src, &mut dst);

        for i in 0..16 {
            let diff = (dst[i].to_f32() - src[i]).abs();
            assert!(diff < 0.01, "diff at {}: {}", i, diff);
        }
    }

    #[test]
    fn test_dup_f16_f32() {
        let src: Vec<f16> = (0..16).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
        let mut dst = vec![0.0f32; 16];

        dup_f16_to_f32(&src, &mut dst);

        for i in 0..16 {
            assert!((dst[i] - src[i].to_f32()).abs() < 1e-6);
        }
    }
}