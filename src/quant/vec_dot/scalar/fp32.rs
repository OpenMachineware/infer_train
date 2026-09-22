/// FP32 vector dot product (scalar implementation)
pub fn vec_dot_fp32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
