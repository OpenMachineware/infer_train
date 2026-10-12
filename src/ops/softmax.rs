// Softmax operator implementation
// Based on llama.cpp's ggml_vec_soft_max_f32

#[cfg(target_arch = "aarch64")]
use super::arm::neon;

/// Scalar softmax: y[i] = exp(x[i] - max) / sum(exp(x[j] - max))
/// Returns the sum for verification
pub fn softmax_f32(x: &[f32], y: &mut [f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    if n == 0 {
        return 0.0;
    }

    // Find max for numerical stability
    let mut max = x[0];
    for &xi in &x[1..] {
        if xi > max {
            max = xi;
        }
    }

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for i in 0..n {
        let val = (x[i] - max).exp();
        y[i] = val;
        sum += val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for yi in y.iter_mut() {
        *yi *= inv_sum;
    }

    sum
}

/// Auto-dispatched softmax using best available implementation
pub fn softmax_f32_dispatch(x: &[f32], y: &mut [f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        unsafe { neon::softmax_f32_neon(x, y) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        softmax_f32(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_scalar() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 4];
        let sum = softmax_f32(&x, &mut y);

        // Verify sum of softmax outputs = 1
        let output_sum: f32 = y.iter().sum();
        assert!((output_sum - 1.0).abs() < 1e-6);

        // Verify values are in (0, 1)
        for &yi in &y {
            assert!(yi > 0.0 && yi < 1.0);
        }

        // Verify larger input gets larger probability
        assert!(y[3] > y[2] && y[2] > y[1] && y[1] > y[0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_neon() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y_scalar = [0.0f32; 4];
        let mut y_neon = [0.0f32; 4];

        let _sum_scalar = softmax_f32(&x, &mut y_scalar);
        let _sum_neon = unsafe { neon::softmax_f32_neon(&x, &mut y_neon) };

        // Verify NEON matches scalar
        for i in 0..4 {
            assert!(
                (y_scalar[i] - y_neon[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                y_scalar[i],
                y_neon[i]
            );
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large values
        let x = [1000.0f32, 1001.0, 1002.0];
        let mut y = [0.0f32; 3];
        softmax_f32(&x, &mut y);

        // Should not overflow
        let output_sum: f32 = y.iter().sum();
        assert!((output_sum - 1.0).abs() < 1e-6);

        // Largest value should have highest probability
        assert!(y[2] > y[1] && y[1] > y[0]);
    }
}
