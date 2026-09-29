use bytemuck::Zeroable;
use half::f16;
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    let ctx = MetalContext::new().unwrap();

    // Test different sizes
    let test_cases = [(4096, 4096), (16384, 4096)];

    for &(m, k) in &test_cases {
        println!("\n=== M={}, K={} ===", m, k);

        // Generate deterministic weights
        let nb = k / 256;
        let mut weights: Vec<BlockQ4K> = Vec::with_capacity(m * nb);
        for row_idx in 0..m {
            for block_idx in 0..nb {
                let mut block = BlockQ4K::zeroed();
                let idx = row_idx * nb + block_idx;
                block.d = f16::to_bits(f16::from_f32(
                    0.5 + 0.001 * (idx % 100) as f32,
                ));
                block.dmin = f16::to_bits(f16::from_f32(0.1));
                // Fill with simple pattern
                for j in 0..12 {
                    block.scales[j] = ((idx + j) % 64) as u8;
                }
                for j in 0..128 {
                    block.qs[j] = ((idx + j) % 256) as u8;
                }
                weights.push(block);
            }
        }

        // Generate input
        let input: Vec<f32> = (0..k).map(|i| 0.1 * ((i % 10) as f32)).collect();

        // Run GPU
        let gpu_result = ctx.mv_q4_k_f32(m, k, &weights, &input).unwrap();

        // Check results
        let sum: f32 = gpu_result.iter().sum();
        let nonzero = gpu_result.iter().filter(|x| **x != 0.0).count();
        let nan_count = gpu_result.iter().filter(|x| x.is_nan()).count();
        let inf_count = gpu_result.iter().filter(|x| x.is_infinite()).count();

        println!(
            "  Result: sum={:.4}, nonzero={}/{}, nan={}, inf={}",
            sum,
            nonzero,
            gpu_result.len(),
            nan_count,
            inf_count
        );

        // Print sample results
        let len = gpu_result.len();
        for i in 0..len.min(5) {
            println!("  result[{}] = {:.4}", i, gpu_result[i]);
        }
        if len > 10 {
            println!("  ...");
            for i in len.saturating_sub(5)..len {
                println!("  result[{}] = {:.4}", i, gpu_result[i]);
            }
        }
    }
}
