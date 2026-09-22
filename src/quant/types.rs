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
// TODO: Add IQ format definitions

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
    }
}
