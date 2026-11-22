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

// Q4_0 interleaved block: 4 rows interleaved
// Layout: 4 scales (8 bytes) + interleaved qs (32 * 4 / 2 = 64 bytes) = 72 bytes
// qs are interleaved with 8-byte chunks
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0x4 {
    pub d: [u16; 4],  // FP16 scales for 4 blocks
    pub qs: [u8; 64], // Interleaved 4-bit values (4 rows * 32 values / 2 bytes per value)
}

// Q4_0 interleaved block: 8 rows interleaved
// Layout: 8 scales (16 bytes) + interleaved qs (32 * 8 / 2 = 128 bytes) = 144 bytes
// qs layout: k * 64 + j * 8 + i where k in 0..1, j in 0..7, i in 0..7
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0x8 {
    pub d: [u16; 8],   // FP16 scales for 8 blocks
    pub qs: [u8; 128], // Interleaved 4-bit values (8 rows * 32 values / 2 bytes per value)
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

// IQ4_XS super-block: 256 values in 136 bytes
// Uses non-linear quantization with grid lookup
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockIQ4XS {
    pub d: u16,            // FP16 super-block scale
    pub scales_h: u16,     // High bits of scales
    pub scales_l: [u8; 4], // Low bits of scales (QK_K/64 = 256/64 = 4)
    pub qs: [u8; 128],     // 4-bit quantized values (QK_K/2 = 256/2 = 128)
}

// ============================================================================
// Repacked block formats for interleaved GEMV (llama.cpp style)
// These pack multiple blocks together for cache-efficient GEMV
// ============================================================================

/// Repacked Q5_K: 8 blocks interleaved
/// Size: 16 + 16 + 96 + 256 + 1024 = 1408 bytes (8 * 176)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ5_Kx8 {
    pub d: [u16; 8],      // 8 scales (16 bytes)
    pub dmin: [u16; 8],   // 8 min scales (16 bytes)
    pub scales: [u8; 96], // Packed scales (96 bytes)
    pub qh: [u8; 256],    // High bits for 8 blocks (256 bytes)
    pub qs: [u8; 1024],   // Low bits for 8 blocks (1024 bytes)
}

/// Repacked Q4_K: 8 blocks interleaved
/// Size: 16 + 16 + 96 + 1024 = 1152 bytes (8 * 144)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_Kx8 {
    pub d: [u16; 8],      // 8 scales (16 bytes)
    pub dmin: [u16; 8],   // 8 min scales (16 bytes)
    pub scales: [u8; 96], // Packed scales (96 bytes)
    pub qs: [u8; 1024],   // Quantized values for 8 blocks (1024 bytes)
}

/// Repacked Q6_K: 8 blocks interleaved
/// Size: 16 + 128 + 1024 + 512 = 1680 bytes (8 * 210)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ6_Kx8 {
    pub d: [u16; 8],       // 8 scales (16 bytes)
    pub scales: [i8; 128], // 8 blocks * 16 scales (128 bytes)
    pub ql: [u8; 1024],    // Low bits for 8 blocks (1024 bytes)
    pub qh: [u8; 512],     // High bits for 8 blocks (512 bytes)
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

impl QuantType {
    /// Get the size of one block in bytes (ggml_type_size)
    pub fn block_size(&self) -> usize {
        match self {
            QuantType::F32 => 4,
            QuantType::F16 | QuantType::BF16 => 2,
            QuantType::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            QuantType::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            QuantType::Q5_0 => 22, // 2 + 2 + 16 + 2 extra = 22 bytes (not defined struct)
            QuantType::Q5_1 => 24, // 2 + 2 + 16 + 4 extra = 24 bytes (not defined struct)
            QuantType::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            QuantType::Q4_K => std::mem::size_of::<BlockQ4_K>(),
            QuantType::Q5_K => std::mem::size_of::<BlockQ5_K>(),
            QuantType::Q6_K => std::mem::size_of::<BlockQ6_K>(),
            QuantType::Q8_K => std::mem::size_of::<BlockQ8_K>(),
            QuantType::I8 => 1,
            QuantType::I16 => 2,
            QuantType::I32 => 4,
            QuantType::I64 => 8,
            QuantType::F64 => 8,
            // For types without defined blocks, use approximate sizes
            QuantType::Q2_K | QuantType::Q3_K => 210, // Same as Q6_K
            // IQ formats: use actual block sizes
            QuantType::IQ4_XS => std::mem::size_of::<BlockIQ4XS>(), // 136 bytes
            QuantType::IQ2_XXS | QuantType::IQ2_XS | QuantType::IQ3_XXS => 2, // TODO: verify
            QuantType::IQ1_S
            | QuantType::IQ4_NL
            | QuantType::IQ3_S
            | QuantType::IQ2_S
            | QuantType::IQ1_M => 210,
            _ => 1,
        }
    }

    /// Get the number of elements per block (ggml_blck_size)
    pub fn block_elements(&self) -> usize {
        match self {
            QuantType::F32 | QuantType::F16 | QuantType::BF16 => 1,
            QuantType::Q4_0 | QuantType::Q4_1 => QK4_0,
            QuantType::Q5_0 | QuantType::Q5_1 => QK5_0,
            QuantType::Q8_0 | QuantType::Q8_1 => QK8_0,
            QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_K => QK_K,
            QuantType::IQ2_XXS
            | QuantType::IQ2_XS
            | QuantType::IQ3_XXS
            | QuantType::IQ1_S
            | QuantType::IQ4_NL
            | QuantType::IQ3_S
            | QuantType::IQ2_S
            | QuantType::IQ4_XS
            | QuantType::IQ1_M => QK_K,
            QuantType::I8
            | QuantType::I16
            | QuantType::I32
            | QuantType::I64
            | QuantType::F64 => 1,
            _ => 1,
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
    Rope { head_dim: usize, max_seq_len: usize, freq_base: f32 },

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
