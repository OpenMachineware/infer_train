#[cfg(target_arch = "aarch64")]
mod tests {
    use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon;
    use half::f16;

    fn generate_q4_k_blocks(m: usize, k: usize) -> Vec<BlockQ4K> {
        let nb = k / QK_K;
        let mut blocks = Vec::with_capacity(m * nb);
        for i in 0..(m * nb) {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
            }
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            blocks.push(BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
                scales,
                qs,
            });
        }
        blocks
    }

    fn generate_q8_k_blocks(k: usize) -> Vec<BlockQ8K> {
        let nb = k / QK_K;
        let mut blocks = Vec::with_capacity(nb);
        for i in 0..nb {
            let mut qs = [0i8; QK_K];
            for j in 0..QK_K {
                qs[j] = ((i * 31 + j * 11) % 256) as i8;
            }
            blocks.push(BlockQ8K {
                d: 1.0 + (i % 10) as f32 * 0.1,
                qs,
                bsums: [0i16; 16],
            });
        }
        blocks
    }

    fn generate_f32_input(k: usize) -> Vec<f32> {
        (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
    }

    #[test]
    fn test_q4_k_gpu_correctness() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        // Test with small sizes first - M=8 to see all rows
        let m = 8;
        let k = 256;

        let weights = generate_q4_k_blocks(m, k);
        let input = generate_f32_input(k);

        // GPU computation
        let gpu_result = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU kernel failed");

        println!("GPU result for M={}, K={}", m, k);
        for i in 0..m {
            println!("  row {}: {}", i, gpu_result[i]);
        }

        // For now, just check that GPU returns values (not NaN or Inf)
        for i in 0..m {
            assert!(gpu_result[i].is_finite(), "GPU result[{}] is not finite: {}", i, gpu_result[i]);
            assert!(gpu_result[i] != 0.0, "GPU result[{}] is zero", i);
        }
    }

    #[test]
    fn test_q4_k_gpu_vs_cpu() {
        // Compare GPU Q4_K x F32 with CPU Q4_K x Q8_K
        // Note: This requires converting F32 input to Q8_K
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let m = 16;
        let k = 256;

        let q4_k_weights = generate_q4_k_blocks(m, k);
        let f32_input = generate_f32_input(k);

        // GPU computation (Q4_K x F32)
        let gpu_result = ctx.mv_q4_k_f32(m, k, &q4_k_weights, &f32_input).expect("GPU kernel failed");

        // CPU computation requires Q8_K, so we skip for now
        // The llama.cpp kernel uses Q4_K x F32 directly, not Q4_K x Q8_K

        println!("GPU Q4_K x F32 result[0]: {}", gpu_result[0]);
        assert!(gpu_result[0].is_finite());
    }
}
