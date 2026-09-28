// Binary operations: ADD, SUB, MUL, DIV
// Matches llama.cpp's binary-ops.cpp template approach
// On macOS, uses vDSP (Accelerate) for F32 - same as llama.cpp

use half::f16;

/// Binary operation trait
pub trait BinaryOp: 'static {
    fn op(a: f32, b: f32) -> f32;
}

pub struct AddOp;
pub struct SubOp;
pub struct MulOp;
pub struct DivOp;

impl BinaryOp for AddOp {
    #[inline(always)]
    fn op(a: f32, b: f32) -> f32 {
        a + b
    }
}

impl BinaryOp for SubOp {
    #[inline(always)]
    fn op(a: f32, b: f32) -> f32 {
        a - b
    }
}

impl BinaryOp for MulOp {
    #[inline(always)]
    fn op(a: f32, b: f32) -> f32 {
        a * b
    }
}

impl BinaryOp for DivOp {
    #[inline(always)]
    fn op(a: f32, b: f32) -> f32 {
        a / b
    }
}

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

/// Scalar binary operation (contiguous arrays)
#[inline]
pub fn binary_scalar<S0, S1, D, OP>(src0: &[S0], src1: &[S1], dst: &mut [D])
where
    S0: FloatType,
    S1: FloatType,
    D: FloatType,
    OP: BinaryOp,
{
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = D::from_f32(OP::op(src0[i].to_f32(), src1[i].to_f32()));
    }
}

/// Scalar binary operation with broadcasting (src1 broadcast across src0)
#[inline]
pub fn binary_broadcast_scalar<S0, S1, D, OP>(
    src0: &[S0],
    src1: &[S1],
    dst: &mut [D],
    ne10: usize,
) where
    S0: FloatType,
    S1: FloatType,
    D: FloatType,
    OP: BinaryOp,
{
    let n = dst.len().min(src0.len());
    let nr = n / ne10;

    for r in 0..nr {
        let offset = r * ne10;
        for i in 0..ne10 {
            let idx = offset + i;
            dst[idx] =
                D::from_f32(OP::op(src0[idx].to_f32(), src1[i].to_f32()));
        }
    }
}

// ============================================================================
// macOS: Use vDSP (Accelerate) - same as llama.cpp
// ============================================================================

#[cfg(target_os = "macos")]
mod vdsp {
    use super::*;
    use std::os::raw::{c_int, c_uint};

    type vDSP_Stride = c_int;
    type vDSP_Length = c_uint;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vDSP_vadd(
            a: *const f32,
            ia: vDSP_Stride,
            b: *const f32,
            ib: vDSP_Stride,
            c: *mut f32,
            ic: vDSP_Stride,
            n: vDSP_Length,
        );
        fn vDSP_vsub(
            a: *const f32,
            ia: vDSP_Stride,
            b: *const f32,
            ib: vDSP_Stride,
            c: *mut f32,
            ic: vDSP_Stride,
            n: vDSP_Length,
        );
        fn vDSP_vmul(
            a: *const f32,
            ia: vDSP_Stride,
            b: *const f32,
            ib: vDSP_Stride,
            c: *mut f32,
            ic: vDSP_Stride,
            n: vDSP_Length,
        );
        fn vDSP_vdiv(
            a: *const f32,
            ia: vDSP_Stride,
            b: *const f32,
            ib: vDSP_Stride,
            c: *mut f32,
            ic: vDSP_Stride,
            n: vDSP_Length,
        );
    }

    pub fn add_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
        let n = dst.len().min(src0.len()).min(src1.len());
        unsafe {
            vDSP_vadd(
                src0.as_ptr(),
                1,
                src1.as_ptr(),
                1,
                dst.as_mut_ptr(),
                1,
                n as vDSP_Length,
            );
        }
        // Handle any remainder (vDSP processes all n elements)
    }

    pub fn sub_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
        let n = dst.len().min(src0.len()).min(src1.len());
        unsafe {
            // vDSP_vsub(a, b) = a - b, so we pass src0 as b, src1 as a to get src0 - src1
            vDSP_vsub(
                src1.as_ptr(),
                1,
                src0.as_ptr(),
                1,
                dst.as_mut_ptr(),
                1,
                n as vDSP_Length,
            );
        }
    }

    pub fn mul_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
        let n = dst.len().min(src0.len()).min(src1.len());
        unsafe {
            vDSP_vmul(
                src0.as_ptr(),
                1,
                src1.as_ptr(),
                1,
                dst.as_mut_ptr(),
                1,
                n as vDSP_Length,
            );
        }
    }

    pub fn div_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
        let n = dst.len().min(src0.len()).min(src1.len());
        unsafe {
            vDSP_vdiv(
                src1.as_ptr(),
                1,
                src0.as_ptr(),
                1,
                dst.as_mut_ptr(),
                1,
                n as vDSP_Length,
            );
        }
    }
}

#[cfg(target_os = "macos")]
pub use vdsp::{add_f32, div_f32, mul_f32, sub_f32};

// ============================================================================
// Non-macOS: Use NEON on aarch64
// ============================================================================

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
mod neon_f32 {
    use super::*;

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
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
pub use neon_f32::{add_f32, div_f32, mul_f32, sub_f32};

// ============================================================================
// Fallback scalar implementations
// ============================================================================

#[cfg(not(target_arch = "aarch64"))]
pub fn add_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    binary_scalar::<f32, f32, f32, AddOp>(src0, src1, dst);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn sub_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    binary_scalar::<f32, f32, f32, SubOp>(src0, src1, dst);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn mul_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    binary_scalar::<f32, f32, f32, MulOp>(src0, src1, dst);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn div_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    binary_scalar::<f32, f32, f32, DivOp>(src0, src1, dst);
}

// ============================================================================
// F16 binary operations - scalar (same as llama.cpp)
// ============================================================================

pub fn add_f16(src0: &[f16], src1: &[f16], dst: &mut [f16]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = f16::from_f32(src0[i].to_f32() + src1[i].to_f32());
    }
}

pub fn sub_f16(src0: &[f16], src1: &[f16], dst: &mut [f16]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = f16::from_f32(src0[i].to_f32() - src1[i].to_f32());
    }
}

pub fn mul_f16(src0: &[f16], src1: &[f16], dst: &mut [f16]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = f16::from_f32(src0[i].to_f32() * src1[i].to_f32());
    }
}

pub fn div_f16(src0: &[f16], src1: &[f16], dst: &mut [f16]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = f16::from_f32(src0[i].to_f32() / src1[i].to_f32());
    }
}

// ============================================================================
// Mixed type operations (F16 + F32)
// ============================================================================

pub fn add_f16_f32(src0: &[f16], src1: &[f32], dst: &mut [f16]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = f16::from_f32(src0[i].to_f32() + src1[i]);
    }
}

pub fn mul_f16_f32(src0: &[f16], src1: &[f32], dst: &mut [f16]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = f16::from_f32(src0[i].to_f32() * src1[i]);
    }
}

pub fn add_f16_f32_to_f32(src0: &[f16], src1: &[f32], dst: &mut [f32]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = src0[i].to_f32() + src1[i];
    }
}

pub fn mul_f16_f32_to_f32(src0: &[f16], src1: &[f32], dst: &mut [f32]) {
    let n = dst.len().min(src0.len()).min(src1.len());
    for i in 0..n {
        dst[i] = src0[i].to_f32() * src1[i];
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32() {
        let src0: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let src1: Vec<f32> = (0..16).map(|i| (i * 2) as f32).collect();
        let mut dst = vec![0.0f32; 16];

        add_f32(&src0, &src1, &mut dst);

        for i in 0..16 {
            assert_eq!(dst[i], src0[i] + src1[i]);
        }
    }

    #[test]
    fn test_mul_f32() {
        let src0: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
        let src1: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let mut dst = vec![0.0f32; 16];

        mul_f32(&src0, &src1, &mut dst);

        for i in 0..16 {
            assert!((dst[i] - src0[i] * src1[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sub_f32() {
        let src0: Vec<f32> = (0..16).map(|i| (i + 10) as f32).collect();
        let src1: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];

        sub_f32(&src0, &src1, &mut dst);

        for i in 0..16 {
            assert_eq!(dst[i], src0[i] - src1[i]);
        }
    }

    #[test]
    fn test_add_f16() {
        let src0: Vec<f16> = (0..16).map(|i| f16::from_f32(i as f32)).collect();
        let src1: Vec<f16> =
            (0..16).map(|i| f16::from_f32((i * 2) as f32)).collect();
        let mut dst = vec![f16::ZERO; 16];

        add_f16(&src0, &src1, &mut dst);

        for i in 0..16 {
            let expected = f16::from_f32(src0[i].to_f32() + src1[i].to_f32());
            assert!((dst[i].to_f32() - expected.to_f32()).abs() < 0.01);
        }
    }
}
