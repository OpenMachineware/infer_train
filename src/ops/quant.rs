// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Quantization data structures (matching llama.cpp GGUF format)

// Block size constants
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
pub const QK_K: usize = 256; // K-quant block size

// Q4_0 block: 32 values, 1 scale, 16 bytes quantized
// Total: 2 + 16 = 18 bytes per 32 values
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: u16,       // FP16 scale
    pub qs: [u8; 16], // 4-bit quantized values (2 per byte)
}

// Q4_1 block: 32 values, 1 scale, 1 minimum, 16 bytes quantized
// Total: 2 + 2 + 16 = 20 bytes per 32 values
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_1 {
    pub d: u16, // FP16 scale
    pub m: u16, // FP16 minimum
    pub qs: [u8; 16],
}

// Q8_0 block: 32 values, 1 scale, 32 bytes quantized
// Total: 2 + 32 = 34 bytes per 32 values
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: u16,       // FP16 scale
    pub qs: [i8; 32], // 8-bit quantized values
}

// Q8_1 block: 32 values, 1 scale, 1 sum, 32 bytes quantized
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_1 {
    pub d: u16, // FP16 scale
    pub s: u16, // FP16 sum
    pub qs: [i8; 32],
}

// K-quant structures (Q2_K through Q8_K)
// These use super-blocks of 256 values

// Q4_K super-block: 256 values in 144 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_K {
    pub d: u16,           // FP16 super-block scale
    pub dmin: u16,        // FP16 minimum scale
    pub scales: [u8; 12], // 12 bytes for scales
    pub qs: [u8; 128],    // 4-bit quantized values
}

// Q5_K super-block: 256 values in 176 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ5_K {
    pub d: u16,           // Super-block scale
    pub dmin: u16,        // Super-block min scale
    pub scales: [u8; 12], // Scales and mins
    pub qh: [u8; 32],     // High bit
    pub qs: [u8; 128],    // Low 4 bits
}

// Q6_K super-block: 256 values in 210 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ6_K {
    pub ql: [u8; 128], // Low bits
    pub qh: [u8; 64],  // High bits
    pub scales: [i8; 16],
    pub d: u16, // FP16 super-block scale
}

// Q8_K super-block: 256 values in 292 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_K {
    pub d: f32,           // FP32 scale
    pub qs: [i8; 256],    // 8-bit quants
    pub bsums: [i16; 16], // Block sums
}

// Helper functions for FP16 conversion
// Use half crate for stable FP16 support
pub fn fp16_to_fp32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

pub fn fp32_to_fp16(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ8_1>(), 36);
        assert_eq!(std::mem::size_of::<BlockQ4_K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ5_K>(), 176);
        assert_eq!(std::mem::size_of::<BlockQ6_K>(), 210);
        assert_eq!(std::mem::size_of::<BlockQ8_K>(), 292);
    }
}
