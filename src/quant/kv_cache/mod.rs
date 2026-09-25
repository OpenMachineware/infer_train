// KV Cache Quantization
// Supports: F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL

mod quant;
mod dequant;

pub use quant::*;
pub use dequant::*;

// Block size for Qn_0/Qn_1 formats
pub const QK: usize = 32;

/// Q8_0 block: 32 int8 values + 1 FP16 scale
/// Memory: 2 + 32 = 34 bytes per 32 floats (25% of FP32)
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ8_0 {
    pub d: u16,        // FP16 scale
    pub qs: [i8; QK],  // 32 int8 quantized values
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

/// IQ4_NL lookup table (from llama.cpp)
pub const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0
];
