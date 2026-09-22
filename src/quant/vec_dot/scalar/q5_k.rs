use crate::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use half::f16;

/// Q5_K × Q8_K vector dot product (scalar implementation)
pub fn vec_dot_q5_k_q8_k(n: usize, x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16::from_bits(x[i].d).to_f32();
        let dmin = y[i].d * f16::from_bits(x[i].dmin).to_f32();

        // TODO: Implement proper scale decoding and high bit handling
        let scales: [u8; 8] = [0; 8];
        let mins: [u8; 8] = [0; 8];

        let min_sum: i32 = (0..8).map(|j| mins[j] as i32 * y[i].bsums[j] as i32).sum();

        let mut isum = 0i32;

        for j in 0..8 {
            for k in 0..16 {
                // Combine 4-bit value with high bit
                let high_bit = ((x[i].qh[j * 2 + k / 8] >> (k % 8)) & 1) as i32;
                let low4 = (x[i].qs[j * 16 + k] & 0x0F) as i32;
                let v = low4 | (high_bit << 4);

                isum += v * y[i].qs[j * 32 + k] as i32 * scales[j] as i32;
            }
        }

        sum += d * isum as f32 - dmin * min_sum as f32;
    }

    sum
}
