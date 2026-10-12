// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Scalar operations: SCALE, ADD1
// Matches llama.cpp's ggml_compute_forward_scale and ggml_compute_forward_add1

#[cfg(target_os = "macos")]
use std::os::raw::{c_int, c_uint};

#[cfg(target_os = "macos")]
type vDSP_Stride = c_int;
#[cfg(target_os = "macos")]
type vDSP_Length = c_uint;

#[cfg(target_os = "macos")]
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
    fn vDSP_vsmul(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vsadd(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
}

/// Scale: dst = src * scale
#[cfg(target_os = "macos")]
pub fn scale_f32(src: &[f32], dst: &mut [f32], scale: f32) {
    let n = dst.len().min(src.len());
    unsafe {
        vDSP_vsmul(
            src.as_ptr(),
            1,
            &scale,
            dst.as_mut_ptr(),
            1,
            n as vDSP_Length,
        );
    }
}

#[cfg(not(target_os = "macos"))]
pub fn scale_f32(src: &[f32], dst: &mut [f32], scale: f32) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = src[i] * scale;
    }
}

/// Scale with bias: dst = src * scale + bias
#[cfg(target_os = "macos")]
pub fn scale_f32_with_bias(
    src: &[f32],
    dst: &mut [f32],
    scale: f32,
    bias: f32,
) {
    let n = dst.len().min(src.len());
    unsafe {
        // vDSP_vsmul then vDSP_vsadd
        vDSP_vsmul(
            src.as_ptr(),
            1,
            &scale,
            dst.as_mut_ptr(),
            1,
            n as vDSP_Length,
        );
        vDSP_vsadd(
            dst.as_mut_ptr(),
            1,
            &bias,
            dst.as_mut_ptr(),
            1,
            n as vDSP_Length,
        );
    }
}

#[cfg(not(target_os = "macos"))]
pub fn scale_f32_with_bias(
    src: &[f32],
    dst: &mut [f32],
    scale: f32,
    bias: f32,
) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = src[i] * scale + bias;
    }
}

/// Add scalar: dst = src + value
#[cfg(target_os = "macos")]
pub fn add1_f32(src: &[f32], dst: &mut [f32], value: f32) {
    let n = dst.len().min(src.len());
    unsafe {
        vDSP_vsadd(
            src.as_ptr(),
            1,
            &value,
            dst.as_mut_ptr(),
            1,
            n as vDSP_Length,
        );
    }
}

#[cfg(not(target_os = "macos"))]
pub fn add1_f32(src: &[f32], dst: &mut [f32], value: f32) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = src[i] + value;
    }
}

/// Multiply-add scalar: dst = src * mul + add
pub fn mad1_f32(src: &[f32], dst: &mut [f32], mul: f32, add: f32) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = src[i] * mul + add;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_f32() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 8];

        scale_f32(&src, &mut dst, 2.0);

        for i in 0..8 {
            assert_eq!(dst[i], src[i] * 2.0);
        }
    }

    #[test]
    fn test_add1_f32() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 8];

        add1_f32(&src, &mut dst, 5.0);

        for i in 0..8 {
            assert_eq!(dst[i], src[i] + 5.0);
        }
    }

    #[test]
    fn test_scale_with_bias() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 8];

        scale_f32_with_bias(&src, &mut dst, 2.0, 1.0);

        for i in 0..8 {
            assert_eq!(dst[i], src[i] * 2.0 + 1.0);
        }
    }
}
