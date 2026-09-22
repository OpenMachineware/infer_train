use crate::quant::types::{BlockQ4_0, BlockQ8_0, QK4_0};
use half::f16;

/// Q4_0 × Q8_0 vector dot product (scalar implementation)
pub fn vec_dot_q4_0_q8_0(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK4_0;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = f16::from_bits(x[i].d).to_f32() * f16::from_bits(y[i].d).to_f32();

        let mut isum = 0i32;
        for j in 0..16 {
            let v0 = (x[i].qs[j] & 0x0F) as i32 - 8;  // Low nibble, bias -8
            let v1 = (x[i].qs[j] >> 4) as i32 - 8;   // High nibble, bias -8

            isum += v0 * y[i].qs[j * 2] as i32;
            isum += v1 * y[i].qs[j * 2 + 1] as i32;
        }

        sum += d * isum as f32;
    }

    sum
}
