use crate::quant::types::{BlockQ2K, BlockQ8K, QK_K};
use half::f16;

/// Q2_K × Q8_K vector dot product (scalar implementation)
pub fn vec_dot_q2_k_q8_k(n: usize, x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16::from_bits(x[i].d).to_f32();
        let dmin = -y[i].d * f16::from_bits(x[i].dmin).to_f32();

        // Process 16 blocks of 16 elements each
        let mut isum = 0i32;
        let mut min_sum = 0i32;

        for j in 0..16 {
            // Extract scale (lower 4 bits) and min (upper 4 bits)
            let scale = (x[i].scales[j] & 0x0F) as i32;
            let min_val = (x[i].scales[j] >> 4) as i32;

            // Q8_K bsums
            min_sum += min_val * y[i].bsums[j] as i32;

            // Unpack 2-bit values for this block
            for k in 0..4 {
                let q2_byte = x[i].qs[j * 4 + k];
                for shift in [0u8, 2, 4, 6] {
                    let q2_val = ((q2_byte >> shift) & 0x03) as i32;
                    let q8_idx = j * 16 + k * 4 + (shift / 2) as usize;
                    isum += q2_val * y[i].qs[q8_idx] as i32 * scale;
                }
            }
        }

        sum += d * isum as f32 + dmin * min_sum as f32;
    }

    sum
}
