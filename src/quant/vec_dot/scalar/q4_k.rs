use crate::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use half::f16;

const KMASK1: u32 = 0x3f3f3f3f;
const KMASK2: u32 = 0x0f0f0f0f;
const KMASK3: u32 = 0x03030303;

/// Decode Q4_K scales from 12-byte format
fn decode_q4_k_scales(scale_bytes: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let utmp: [u32; 3] = [
        u32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]),
        u32::from_le_bytes([scale_bytes[4], scale_bytes[5], scale_bytes[6], scale_bytes[7]]),
        u32::from_le_bytes([scale_bytes[8], scale_bytes[9], scale_bytes[10], scale_bytes[11]]),
    ];

    let mins8_0 = utmp[1] & KMASK1;
    let mins8_1 = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);

    let scales_0 = utmp[0] & KMASK1;
    let scales_1 = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);

    let scales = unsafe {
        let mut arr = [0u8; 8];
        std::ptr::copy_nonoverlapping(&scales_0 as *const u32 as *const u8, arr.as_mut_ptr(), 4);
        std::ptr::copy_nonoverlapping(&scales_1 as *const u32 as *const u8, arr.as_mut_ptr().add(4), 4);
        arr
    };

    let mins = unsafe {
        let mut arr = [0u8; 8];
        std::ptr::copy_nonoverlapping(&mins8_0 as *const u32 as *const u8, arr.as_mut_ptr(), 4);
        std::ptr::copy_nonoverlapping(&mins8_1 as *const u32 as *const u8, arr.as_mut_ptr().add(4), 4);
        arr
    };

    (scales, mins)
}

/// Q4_K × Q8_K vector dot product (scalar implementation)
pub fn vec_dot_q4_k_q8_k(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16::from_bits(x[i].d).to_f32();
        let dmin = y[i].d * f16::from_bits(x[i].dmin).to_f32();

        let (scales, mins) = decode_q4_k_scales(&x[i].scales);

        // Calculate min correction
        let min_sum: i32 = (0..8).map(|j| mins[j] as i32 * y[i].bsums[j] as i32).sum();

        let mut isum = 0i32;

        // Process 8 blocks of 32 elements each
        for j in 0..4 {
            // Low nibble
            for k in 0..16 {
                let v = (x[i].qs[j * 32 + k] & 0x0F) as i32 - 8;
                isum += v * y[i].qs[j * 64 + k] as i32 * scales[j * 2] as i32;
            }

            // High nibble
            for k in 0..16 {
                let v = (x[i].qs[j * 32 + k] >> 4) as i32 - 8;
                isum += v * y[i].qs[j * 64 + 32 + k] as i32 * scales[j * 2 + 1] as i32;
            }
        }

        sum += d * isum as f32 - dmin * min_sum as f32;
    }

    sum
}
