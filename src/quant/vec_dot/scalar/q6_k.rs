use crate::quant::types::{BlockQ6K, BlockQ8K, QK_K};
use half::f16;

/// Q6_K × Q8_K vector dot product (scalar implementation)
pub fn vec_dot_q6_k_q8_k(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16::from_bits(x[i].d).to_f32();

        let mut isum = 0i32;

        // Process 16 blocks of 16 elements each
        for j in 0..16 {
            for k in 0..8 {
                // Combine lower 4 bits and upper 2 bits
                let lower4 = x[i].ql[j * 8 + k] as i32;
                let upper2 = (x[i].qh[j * 4 + k / 2] >> (4 * (k % 2))) as i32 & 0x03;
                let v = lower4 | (upper2 << 4);

                isum += v * y[i].qs[j * 16 + k] as i32 * x[i].scales[j] as i32;
                isum += v * y[i].qs[j * 16 + k + 8] as i32 * x[i].scales[j] as i32;
            }
        }

        sum += d * isum as f32;
    }

    sum
}
