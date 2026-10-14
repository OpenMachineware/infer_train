// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// NEON DUP/CPY operations for type conversion

use half::f16;

/// NEON F32 -> F16 conversion
#[target_feature(enable = "neon")]
pub unsafe fn dup_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    let n = dst.len().min(src.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        // Convert using scalar (NEON f16 intrinsics are unstable)
        for j in 0..4 {
            dst[offset + j] = f16::from_f32(src[offset + j]);
        }
    }

    for i in (chunks * 4)..n {
        dst[i] = f16::from_f32(src[i]);
    }
}

/// NEON F16 -> F32 conversion
#[target_feature(enable = "neon")]
pub unsafe fn dup_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    let n = dst.len().min(src.len());
    let chunks = n / 4;

    for i in 0..chunks {
        let offset = i * 4;
        for j in 0..4 {
            dst[offset + j] = src[offset + j].to_f32();
        }
    }

    for i in (chunks * 4)..n {
        dst[i] = src[i].to_f32();
    }
}
