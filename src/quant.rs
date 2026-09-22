use bytemuck::{Pod, Zeroable};

/// FP16 stored as u16 for Pod compatibility
pub type F16 = u16;

/// Convert F16 to f32
#[inline]
pub fn f16_to_f32(h: F16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Convert f32 to F16
#[inline]
pub fn f32_to_f16(f: f32) -> F16 {
    half::f16::from_f32(f).to_bits()
}

/// Q2_K block: 2-bit quantization with scale and min
/// 16 blocks of 16 elements each
/// Total: 2*sizeof(ggml_half) + QK_K/16 + QK_K/4 = 4 + 16 + 64 = 84 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ2K {
    pub scales: [u8; 16],    // scales and mins, quantized with 4 bits
    pub qs: [u8; 64],        // quants (2 bits per element, 256/4=64 bytes)
    pub d: F16,              // super-block scale for quantized scales
    pub dmin: F16,           // super-block scale for quantized mins
}

/// Q3_K block: 3-bit quantization
/// 16 blocks of 16 elements each
/// Total: sizeof(ggml_half) + QK_K/4 + QK_K/8 + 12 = 2 + 64 + 32 + 12 = 110 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],     // quants - high bit
    pub qs: [u8; 64],        // quants - low 2 bits
    pub scales: [u8; 12],    // scales, quantized with 6 bits
    pub d: F16,              // super-block scale
}

/// Q4_K block: 4-bit quantization with scale and min
/// 8 blocks of 32 elements each
/// Total: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 = 4 + 12 + 128 = 144 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ4K {
    pub d: F16,              // super-block scale for quantized scales
    pub dmin: F16,           // super-block scale for quantized mins
    pub scales: [u8; 12],    // scales and mins, quantized with 6 bits
    pub qs: [u8; 128],       // 4-bit quants
}

/// Q5_K block: 5-bit quantization with scale and min
/// 8 blocks of 32 elements each
/// Total: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8 = 4 + 12 + 128 + 32 = 176 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ5K {
    pub d: F16,              // super-block scale for quantized scales
    pub dmin: F16,           // super-block scale for quantized mins
    pub scales: [u8; 12],    // scales and mins, quantized with 6 bits
    pub qh: [u8; 32],        // quants, high bit
    pub qs: [u8; 128],       // quants, low 4 bits
}

/// Q6_K block: 6-bit quantization
/// 16 blocks of 16 elements each
/// Total: sizeof(ggml_half) + QK_K/16 + 3*QK_K/4 = 2 + 16 + 192 = 210 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ6K {
    pub ql: [u8; 128],       // quants, lower 4 bits
    pub qh: [u8; 64],        // quants, upper 2 bits
    pub scales: [i8; 16],    // scales, quantized with 8 bits
    pub d: F16,              // super-block scale
}

/// Q8_K block: intermediate quantization format
/// Used for dot products and intermediate quantization
/// Total: sizeof(float) + QK_K + QK_K/16*sizeof(int16_t) = 4 + 256 + 32 = 292 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BlockQ8K {
    pub d: f32,              // delta
    pub qs: [i8; 256],       // quants
    pub bsums: [i16; 16],    // sum of quants in groups of 16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
    }
}
