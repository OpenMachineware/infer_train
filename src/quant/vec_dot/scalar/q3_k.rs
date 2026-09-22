use crate::quant::types::{BlockQ3K, BlockQ8K, QK_K};
use half::f16;

/// Q3_K × Q8_K vector dot product (scalar implementation)
pub fn vec_dot_q3_k_q8_k(n: usize, x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16::from_bits(x[i].d).to_f32();

        // Decode scales from 12-byte format
        // TODO: Implement proper scale decoding
        let scales: [i8; 16] = [0; 16];

        let mut isum = 0i32;

        // Process 16 blocks of 16 elements each
        for j in 0..16 {
            // Unpack 3-bit values (2 bits from qs, 1 bit from hmask)
            for k in 0..4 {
                let q3_byte = x[i].qs[j * 4 + k];
                let hmask_byte = x[i].hmask[j / 2];

                for shift in [0u8, 2, 4, 6] {
                    let low2 = ((q3_byte >> shift) & 0x03) as i32;
                    let high1 = ((hmask_byte >> (j % 2 * 8 + k * 2 + (shift / 2) as usize)) & 1) as i32;
                    let q3_val = low2 | (high1 << 2);

                    let q8_idx = j * 16 + k * 4 + (shift / 2) as usize;
                    isum += q3_val * y[i].qs[q8_idx] as i32 * scales[j] as i32;
                }
            }
        }

        sum += d * isum as f32;
    }

    sum
}
