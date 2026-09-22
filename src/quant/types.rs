use bytemuck::{Pod, Zeroable};

pub const QK4_0: usize = 32;  // Block size for Q4_0
pub const QK8_0: usize = 32;  // Block size for Q8_0
pub const QK_K: usize = 256;  // Block size for K-quant

// ===== Basic quantization formats =====

/// Q4_0 block: 4-bit quantization with scale
/// 32 elements per block, 18 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ4_0 {
    pub d: u16,        // FP16 scale
    pub qs: [u8; 16],  // 4-bit quants (32 elements packed into 16 bytes)
}

/// Q4_1 block: 4-bit quantization with scale and min
/// 32 elements per block, 20 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ4_1 {
    pub d: u16,        // FP16 scale
    pub m: u16,        // FP16 min
    pub qs: [u8; 16],  // 4-bit quants
}

/// Q5_0 block: 5-bit quantization with scale
/// 32 elements per block, 22 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ5_0 {
    pub d: u16,         // FP16 scale
    pub qh: [u8; 4],    // High bit for 32 elements
    pub qs: [u8; 16],   // Low 4 bits
}

/// Q5_1 block: 5-bit quantization with scale and min
/// 32 elements per block, 24 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ5_1 {
    pub d: u16,         // FP16 scale
    pub m: u16,         // FP16 min
    pub qh: [u8; 4],    // High bit for 32 elements
    pub qs: [u8; 16],   // Low 4 bits
}

/// Q8_0 block: 8-bit quantization with scale
/// 32 elements per block, 34 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ8_0 {
    pub d: u16,         // FP16 scale
    pub qs: [i8; 32],   // 8-bit quants
}

/// Q8_1 block: 8-bit quantization with scale and sum
/// 32 elements per block, 36 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ8_1 {
    pub d: u16,         // FP16 scale
    pub s: u16,         // FP16 sum
    pub qs: [i8; 32],   // 8-bit quants
}

// ===== K-quant formats =====

/// Q2_K block: 2-bit quantization with scale and min
/// 256 elements per block, 84 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ2K {
    pub scales: [u8; 16],  // Scales and mins (4 bits each)
    pub qs: [u8; 64],      // 2-bit quants
    pub d: u16,            // Super-block scale
    pub dmin: u16,         // Super-block min scale
}

/// Q3_K block: 3-bit quantization
/// 256 elements per block, 110 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],   // High bit mask
    pub qs: [u8; 64],      // Low 2 bits
    pub scales: [u8; 12],  // Scales
    pub d: u16,            // Super-block scale
}

/// Q4_K block: 4-bit quantization with scale and min
/// 256 elements per block, 144 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ4K {
    pub d: u16,            // Super-block scale
    pub dmin: u16,         // Super-block min scale
    pub scales: [u8; 12],  // Scales and mins
    pub qs: [u8; 128],     // 4-bit quants
}

/// Q5_K block: 5-bit quantization with scale and min
/// 256 elements per block, 176 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ5K {
    pub d: u16,            // Super-block scale
    pub dmin: u16,         // Super-block min scale
    pub scales: [u8; 12],  // Scales and mins
    pub qh: [u8; 32],      // High bit
    pub qs: [u8; 128],     // Low 4 bits
}

/// Q6_K block: 6-bit quantization
/// 256 elements per block, 210 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ6K {
    pub ql: [u8; 128],     // Lower 4 bits
    pub qh: [u8; 64],      // Upper 2 bits
    pub scales: [i8; 16],  // Scales
    pub d: u16,            // Super-block scale
}

/// Q8_K block: 8-bit intermediate quantization
/// 256 elements per block, 292 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ8K {
    pub d: f32,            // Scale
    pub qs: [i8; 256],     // 8-bit quants
    pub bsums: [i16; 16],  // Block sums
}

// ===== IQ formats =====

/// IQ1_S block: 1.5625 bpw
/// 256 elements per block, 42 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ1S {
    pub d: u16,          // FP16 scale
    pub qs: [u8; 32],    // Grid indices (QK_K/8 = 256/8 = 32)
    pub qh: [u16; 8],    // High bits (QK_K/32 = 256/32 = 8)
}

/// IQ1_M block: 1.75 bpw
/// 256 elements per block, 56 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ1M {
    pub qs: [u8; 32],     // Grid indices, low 8 bits
    pub qh: [u8; 16],     // Grid indices, high 3 bits + grid shift bit
    pub scales: [u8; 8],  // 3-bit block scales
}

/// IQ2_XXS block: 2.0625 bpw
/// 256 elements per block, 66 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ2XXS {
    pub d: u16,        // FP16 scale
    pub qs: [u16; 32], // QK_K/8 = 256/8 = 32
}

/// IQ2_XS block: 2.3125 bpw
/// 256 elements per block, 74 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ2XS {
    pub d: u16,        // FP16 scale
    pub qs: [u16; 32], // QK_K/8 = 256/8 = 32
    pub scales: [u8; 8], // QK_K/32 = 256/32 = 8
}

/// IQ2_S block: 2.5625 bpw
/// 256 elements per block, 82 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ2S {
    pub d: u16,         // FP16 scale
    pub qs: [u8; 64],   // QK_K/4 = 256/4 = 64
    pub qh: [u8; 8],    // QK_K/32 = 256/32 = 8
    pub scales: [u8; 8], // QK_K/32 = 256/32 = 8
}

/// IQ3_XXS block: 3.0625 bpw
/// 256 elements per block, 98 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ3XXS {
    pub d: u16,       // FP16 scale
    pub qs: [u8; 96], // 3*QK_K/8 = 3*256/8 = 96
}

/// IQ3_S block: 3.4375 bpw
/// 256 elements per block, 110 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ3S {
    pub d: u16,         // FP16 scale
    pub qs: [u8; 64],   // QK_K/4 = 256/4 = 64
    pub qh: [u8; 8],    // QK_K/32 = 256/32 = 8
    pub signs: [u8; 32], // QK_K/8 = 256/8 = 32
    pub scales: [u8; 4], // QK_K/64 = 256/64 = 4
}

/// IQ4_NL block: 4-bit non-linear
/// 32 elements per block, 18 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ4NL {
    pub d: u16,       // FP16 scale
    pub qs: [u8; 16], // QK4_NL/2 = 32/2 = 16
}

/// IQ4_XS block: 4.0625 bpw
/// 256 elements per block, 136 bytes total
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockIQ4XS {
    pub d: u16,         // FP16 scale
    pub scales_h: u16,  // High bits of scales
    pub scales_l: [u8; 4], // QK_K/64 = 256/64 = 4
    pub qs: [u8; 128],  // QK_K/2 = 256/2 = 128
}

// ===== TQ formats =====

/// TQ1_0 block: 1.5625 bpw
/// 256 elements per block
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockTQ1_0 {
    pub d: u16,       // FP16 scale
    pub qs: [u8; 32], // QK_K/8 = 256/8 = 32
}

/// TQ2_0 block: 2 bpw
/// 256 elements per block
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockTQ2_0 {
    pub d: u16,       // FP16 scale
    pub qs: [u8; 64], // QK_K/4 = 256/4 = 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockQ5_0>(), 22);
        assert_eq!(std::mem::size_of::<BlockQ5_1>(), 24);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ8_1>(), 36);
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
        assert_eq!(std::mem::size_of::<BlockIQ1S>(), 50);  // 2 + 32 + 16
        assert_eq!(std::mem::size_of::<BlockIQ1M>(), 56);
        assert_eq!(std::mem::size_of::<BlockIQ2XXS>(), 66);
        assert_eq!(std::mem::size_of::<BlockIQ2XS>(), 74);
        assert_eq!(std::mem::size_of::<BlockIQ2S>(), 82);
        assert_eq!(std::mem::size_of::<BlockIQ3XXS>(), 98);
        assert_eq!(std::mem::size_of::<BlockIQ3S>(), 110);
        assert_eq!(std::mem::size_of::<BlockIQ4NL>(), 18);
        assert_eq!(std::mem::size_of::<BlockIQ4XS>(), 136);
    }
}
