// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Data types for tensor and quantization

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

// Quantization type (ggml_type values)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
}

impl TryFrom<i32> for QuantType {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(QuantType::F32),
            1 => Ok(QuantType::F16),
            2 => Ok(QuantType::Q4_0),
            3 => Ok(QuantType::Q4_1),
            4 => Ok(QuantType::Q4_2),
            5 => Ok(QuantType::Q4_3),
            6 => Ok(QuantType::Q5_0),
            7 => Ok(QuantType::Q5_1),
            8 => Ok(QuantType::Q8_0),
            9 => Ok(QuantType::Q8_1),
            10 => Ok(QuantType::Q2_K),
            11 => Ok(QuantType::Q3_K),
            12 => Ok(QuantType::Q4_K),
            13 => Ok(QuantType::Q5_K),
            14 => Ok(QuantType::Q6_K),
            15 => Ok(QuantType::Q8_K),
            16 => Ok(QuantType::IQ2_XXS),
            17 => Ok(QuantType::IQ2_XS),
            18 => Ok(QuantType::IQ3_XXS),
            19 => Ok(QuantType::IQ1_S),
            20 => Ok(QuantType::IQ4_NL),
            21 => Ok(QuantType::IQ3_S),
            22 => Ok(QuantType::IQ2_S),
            23 => Ok(QuantType::IQ4_XS),
            24 => Ok(QuantType::I8),
            25 => Ok(QuantType::I16),
            26 => Ok(QuantType::I32),
            27 => Ok(QuantType::I64),
            28 => Ok(QuantType::F64),
            29 => Ok(QuantType::IQ1_M),
            30 => Ok(QuantType::BF16),
            31 => Ok(QuantType::Q4_0_4_4),
            32 => Ok(QuantType::Q4_0_4_8),
            33 => Ok(QuantType::Q4_0_8_8),
            _ => Err(format!("Unknown quant type: {}", value)),
        }
    }
}

// Core operators
#[derive(Debug, Clone)]
pub enum CoreOperator {
    /// RMS normalization
    RmsNorm { hidden_size: usize, eps: f32 },

    /// Linear layer (supports quantization)
    Linear { in_features: usize, out_features: usize, quant_type: QuantType },

    /// RoPE positional encoding
    Rope { head_dim: usize, max_seq_len: usize },

    /// Flash Attention
    FlashAttention {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    },

    /// SwiGLU activation
    SwiGLU { hidden_size: usize, intermediate_size: usize },

    /// MoE router
    MoERouter { n_experts: usize, top_k: usize },

    /// MoE expert
    MoEExpert {
        expert_id: usize,
        in_features: usize,
        out_features: usize,
        quant_type: QuantType,
    },
}

// Operator execution error
#[derive(Debug)]
pub enum OpError {
    NotImplemented(String),
    InvalidInput(String),
    DeviceError(String),
}

// Helper functions for FP16 conversion
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
