use half::f16;

/// FP16 vector dot product (scalar implementation)
pub fn vec_dot_fp16(a: &[f16], b: &[f16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.to_f32() * y.to_f32())
        .sum()
}
