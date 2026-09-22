pub mod arm;

/// CPU kernel trait for quantized matrix multiplication
pub trait QuantMatmul {
    /// Quantized weights x Quantized activations
    /// x: quantized weights (Qn_K format)
    /// y: quantized activations (Q8_K format)
    /// returns: dot product result
    fn vec_dot_qk_q8k(n: usize, x: &[u8], y: &[u8]) -> f32;
}
