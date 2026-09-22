use crate::quant::types::{BlockQ8_0, QK8_0};
use half::f16;

/// Q8_0 × Q8_0 vector dot product (scalar implementation)
pub fn vec_dot_q8_0_q8_0(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK8_0;
    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = f16::from_bits(x[i].d).to_f32() * f16::from_bits(y[i].d).to_f32();

        let isum: i32 = x[i].qs.iter().zip(y[i].qs.iter())
            .map(|(&a, &b)| a as i32 * b as i32)
            .sum();

        sum += d * isum as f32;
    }

    sum
}
