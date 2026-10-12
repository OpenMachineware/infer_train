// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// DUP/CPY operations: copy with type conversion

use half::f16;

#[cfg(target_arch = "aarch64")]
use super::arm::neon;

/// Type conversion trait
pub trait FloatType: Copy + Clone {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl FloatType for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl FloatType for f16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
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

// CPY is just an alias for DUP
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
}
