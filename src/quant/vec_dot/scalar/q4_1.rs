use crate::quant::types::{BlockQ4_1, BlockQ8_1, QK4_0};
use half::f16;

/// Q4_1 × Q8_1 vector dot product (scalar implementation)
/// Note: Q4_1 has scale and min, Q8_1 has scale and sum
pub fn vec_dot_q4_1_q8_1(n: usize, x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    let nb = n / QK4_0;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = f16::from_bits(x[i].d).to_f32() * f16::from_bits(y[i].d).to_f32();
        let m = f16::from_bits(x[i].m).to_f32() * f16::from_bits(y[i].s).to_f32();

        let mut isum = 0i32;
        for j in 0..16 {
            let v0 = (x[i].qs[j] & 0x0F) as i32;
            let v1 = (x[i].qs[j] >> 4) as i32;

            isum += v0 * y[i].qs[j * 2] as i32;
            isum += v1 * y[i].qs[j * 2 + 1] as i32;
        }

        sum += d * isum as f32 + m;
    }

    sum
}
