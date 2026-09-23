#[cfg(target_arch = "aarch64")]
mod tests {
    use infer_train::quant::types::BlockQ4_0;
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use half::f16;

    fn generate_q4_0_weights(m: usize, k: usize) -> Vec<BlockQ4_0> {
        let nb = k / 32;
        let mut weights = Vec::with_capacity(m * nb);
        for i in 0..(m * nb) {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            weights.push(BlockQ4_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            });
        }
        weights
    }

    fn generate_f32_input(k: usize) -> Vec<f32> {
        (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
    }

    #[test]
    fn test_q4_0_gpu_correctness() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        // Test with M=8 to see all rows
        let m = 8;
        let k = 256;

        let weights = generate_q4_0_weights(m, k);
        let input = generate_f32_input(k);

        let gpu_result = ctx.mv_q4_0_f32(m, k, &weights, &input).expect("GPU kernel failed");

        println!("Q4_0 GPU result for M={}, K={}", m, k);
        for i in 0..m {
            println!("  row {}: {}", i, gpu_result[i]);
        }

        // Check all rows have values
        for i in 0..m {
            assert!(gpu_result[i].is_finite(), "GPU result[{}] is not finite: {}", i, gpu_result[i]);
            assert!(gpu_result[i] != 0.0, "GPU result[{}] is zero", i);
        }
    }
}
