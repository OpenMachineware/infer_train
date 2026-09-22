use half::bf16;

/// BF16 vector dot product (scalar implementation)
pub fn vec_dot_bf16(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.to_f32() * y.to_f32())
        .sum()
}
