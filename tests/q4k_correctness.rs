#[cfg(target_arch = "aarch64")]
mod tests {
    use infer_train::quant::{BlockQ4K, BlockQ8K, f16_to_f32, f32_to_f16};
    use infer_train::ops::cpu::arm::vec_dot_q4k_q8k;

    /// Generate test data (same as llama.cpp)
    fn generate_test_data(offset: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.1 + 2.0 * ((i as f32) + offset).cos())
            .collect()
    }

    /// Simple Q8_K quantization (for testing)
    fn quantize_q8k_simple(data: &[f32]) -> BlockQ8K {
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];

        // Find max absolute value
        let max_val = data.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| a.max(b));
        let d = max_val / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Quantize
        for (i, &val) in data.iter().enumerate() {
            let q = (val * id).round().clamp(-127.0, 127.0) as i8;
            qs[i] = q;
        }

        // Calculate block sums
        for i in 0..16 {
            let mut sum = 0i32;
            for j in 0..16 {
                sum += qs[i * 16 + j] as i32;
            }
            bsums[i] = sum as i16;
        }

        BlockQ8K { d, qs, bsums }
    }

    /// Simple Q4_K quantization (for testing)
    /// Note: This is a simplified version for testing, not production-ready
    fn quantize_q4k_simple(data: &[f32]) -> BlockQ4K {
        let mut qs = [0u8; 128];
        let mut scales = [0u8; 12];

        // Simple quantization: each byte packs two 4-bit values
        // Scale: use global scale
        let max_val = data.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| a.max(b));
        let d = max_val / 7.0;

        for (i, chunk) in data.chunks(2).enumerate() {
            let v0 = if d != 0.0 { ((chunk[0] / d).round().clamp(-8.0, 7.0) as i32 + 8) as u8 } else { 8 };
            let v1 = if chunk.len() > 1 && d != 0.0 {
                ((chunk[1] / d).round().clamp(-8.0, 7.0) as i32 + 8) as u8
            } else {
                8
            };
            qs[i] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
        }

        // Set scales (simplified: all same scale)
        scales[0] = 8; // Scale value
        for i in 1..8 {
            scales[i] = 8;
        }

        BlockQ4K {
            d: f32_to_f16(d),
            dmin: f32_to_f16(0.0),
            scales,
            qs,
        }
    }

    /// Reference dot product (scalar)
    fn dot_product_ref(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn test_q4k_q8k_correctness() {
        // Generate test data
        let src = generate_test_data(0.0, 256);
        let act = generate_test_data(1.0, 256);

        // Quantize
        let q4k = quantize_q4k_simple(&src);
        let q8k = quantize_q8k_simple(&act);

        // Compute reference dot product
        let ref_result = dot_product_ref(&src, &act);

        // Compute using NEON kernel
        let neon_result = unsafe {
            vec_dot_q4k_q8k(256, &[q4k], &[q8k])
        };

        println!("Reference: {}", ref_result);
        println!("NEON:      {}", neon_result);
        println!("Diff:      {}", (ref_result - neon_result).abs());

        // Allow 10% error (quantization introduces error)
        let relative_error = (ref_result - neon_result).abs() / ref_result.abs();
        assert!(relative_error < 0.1, "Relative error too large: {}", relative_error);
    }
}
