// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// KV Cache quantization block structures
// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Jia Liu

// Block size for Qn_0/Qn_1 formats
pub const QK: usize = 32;

/// Q8_0 block: 32 int8 values + 1 FP16 scale
/// Memory: 2 + 32 = 34 bytes per 32 floats (25% of FP32)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ8_0 {
    pub d: u16,       // FP16 scale
    pub qs: [i8; QK], // 32 int8 quantized values
}

/// Q8_1 block: 32 int8 values + FP16 scale + FP16 sum
/// Memory: 2 + 2 + 32 = 36 bytes per 32 floats
/// Used for Q4_1 and Q5_1 vec_dot
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ8_1 {
    pub d: u16,       // FP16 scale
    pub s: u16,       // FP16 sum of quantized values
    pub qs: [i8; QK], // 32 int8 quantized values
}

/// Q4_0 block: 16 nibbles + 1 FP16 scale
/// Memory: 2 + 16 = 18 bytes per 32 floats (12.5% of FP32)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ4_0 {
    pub d: u16,           // FP16 scale
    pub qs: [u8; QK / 2], // 16 bytes, each byte holds 2 4-bit values
}

/// Q4_1 block: 16 nibbles + 1 FP16 scale + 1 FP16 min
/// Memory: 4 + 16 = 20 bytes per 32 floats (14% of FP32)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ4_1 {
    pub d: u16,           // FP16 scale
    pub m: u16,           // FP16 min (bias)
    pub qs: [u8; QK / 2], // 16 bytes, each byte holds 2 4-bit values
}

/// Q5_0 block: 16 nibbles + 4 bytes high bits + 1 FP16 scale
/// Memory: 2 + 4 + 16 = 22 bytes per 32 floats (16% of FP32)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ5_0 {
    pub d: u16,           // FP16 scale
    pub qh: [u8; 4],      // 5th bit for each of 32 values
    pub qs: [u8; QK / 2], // 16 bytes, each byte holds 2 4-bit values
}

/// Q5_1 block: 16 nibbles + 4 bytes high bits + 1 FP16 scale + 1 FP16 min
/// Memory: 4 + 4 + 16 = 24 bytes per 32 floats (17% of FP32)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ5_1 {
    pub d: u16,           // FP16 scale
    pub m: u16,           // FP16 min (bias)
    pub qh: [u8; 4],      // 5th bit for each of 32 values
    pub qs: [u8; QK / 2], // 16 bytes, each byte holds 2 4-bit values
}

/// IQ4_NL block: 16 nibbles with non-linear lookup table
/// Memory: 2 + 16 = 18 bytes per 32 floats (12.5% of FP32)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockIQ4NL {
    pub d: u16,           // FP16 scale
    pub qs: [u8; QK / 2], // 16 bytes, each byte holds 2 4-bit indices
}

pub const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0,
    38.0, 53.0, 69.0, 89.0, 113.0,
];

/// IQ4_NL lookup table as int8 (for NEON TBL instruction)
pub const KVALUES_IQ4NL_I8: [i8; 16] =
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

// ===== K-quant formats for KV cache (Metal layout compatible) =====
// These structures must match the Metal shader layout exactly

/// Block size for K-quant formats
pub const QK_K: usize = 256;

/// Q2_K block: 2-bit quantization with scale and min
/// 256 elements per block, 84 bytes total
/// Layout matches Metal shader: d, dmin, scales, qs
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ2K_KV {
    pub d: u16,           // Super-block scale (offset 0)
    pub dmin: u16,        // Super-block min scale (offset 2)
    pub scales: [u8; 16], // Scales and mins (4 bits each) (offset 4)
    pub qs: [u8; 64],     // 2-bit quants (offset 20)
}

/// Q3_K block: 3-bit quantization
/// 256 elements per block, 110 bytes total
/// Layout matches Metal shader: d, hmask, qs, scales
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ3K_KV {
    pub d: u16,          // Super-block scale (offset 0)
    pub hmask: [u8; 32], // High bit mask (offset 2)
    pub qs: [u8; 64],    // Low 2 bits (offset 34)
    pub scales: [i8; 12], // Scales (offset 98)
                         // Note: there's 2 bytes padding at the end to make it 110 bytes
}

/// Q4_K block: 4-bit quantization with scale and min
/// 256 elements per block, 144 bytes total
/// Layout matches Metal shader: d, dmin, scales, qs
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ4K_KV {
    pub d: u16,           // Super-block scale (offset 0)
    pub dmin: u16,        // Super-block min scale (offset 2)
    pub scales: [u8; 12], // Scales and mins (offset 4)
    pub qs: [u8; 128],    // 4-bit quants, packed 2 per byte (offset 16)
}

/// Q5_K block: 5-bit quantization with scale and min
/// 256 elements per block, 176 bytes total
/// Layout matches Metal shader: d, dmin, scales, qh, qs
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ5K_KV {
    pub d: u16,           // Super-block scale (offset 0)
    pub dmin: u16,        // Super-block min scale (offset 2)
    pub scales: [u8; 12], // Scales and mins (offset 4)
    pub qh: [u8; 32],     // High bits (offset 16)
    pub qs: [u8; 128],    // Low 4 bits (offset 48)
}

/// Q6_K block: 6-bit quantization
/// 256 elements per block, 210 bytes total
/// Layout matches Metal shader: d, scales, ql, qh
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ6K_KV {
    pub d: u16,           // Super-block scale (offset 0)
    pub scales: [i8; 16], // Scales (offset 2)
    pub ql: [u8; 128],    // Lower 4 bits (offset 18)
    pub qh: [u8; 64],     // Upper 2 bits (offset 146)
}

// Verify sizes at compile time
const _: () = {
    assert!(std::mem::size_of::<BlockQ2K_KV>() == 84);
    assert!(std::mem::size_of::<BlockQ3K_KV>() == 110);
    assert!(std::mem::size_of::<BlockQ4K_KV>() == 144);
    assert!(std::mem::size_of::<BlockQ5K_KV>() == 176);
    assert!(std::mem::size_of::<BlockQ6K_KV>() == 210);
};
