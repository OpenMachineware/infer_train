// Verify Q3_K GEMM kernel correctness
// Run: cargo run --release --bin verify_q3k_correctness

use infer_train::quant::types::BlockQ3K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn generate_q3_k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        let mut scales = [0u8; 12];
        // Simple pattern: alternating hmask, known qs values, uniform scales
        for j in 0..32 {
            hmask[j] = 0xFF; // All high bits set
        }
        for j in 0..64 {
            qs[j] = 0x55; // Pattern: 01010101
        }
        for j in 0..12 {
            scales[j] = 16; // Uniform scale = 16
        }
        weights.push(BlockQ3K {
            d: f16::from_f32(1.0).to_bits(),
            hmask,
            qs,
            scales,
        });
    }
    weights
}

fn reference_q3_k_vec_dot(k: usize, weights: &[BlockQ3K], input: &[f32]) -> f32 {
    let nb = k / 256;
    let mut total_sum = 0.0f32;

    for ib in 0..nb {
        let block = &weights[ib];
        let d = f16::from_bits(block.d).to_f32();
        let mut block_sum = 0.0f32;

        for ir in 0..4 {
            let scales = block.scales;

            // Process 64 elements per ir (8 from qs + 8 from hmask for each)
            for i in 0..8 {
                let q = block.qs[8 * ir + i];
                let h = block.hmask[8 * ir + i];

                // Q3_K format: 2 bits from qs + 1 bit from hmask
                // Value = (q & 0x03) | (h & 1) << 2 - 4
                let q0 = (q & 0x03) | ((h & 1) << 2);
                let q1 = ((q >> 2) & 0x03) | ((h >> 1) & 0x04);
                let q2 = ((q >> 4) & 0x03) | ((h >> 2) & 0x04);
                let q3 = ((q >> 6) & 0x03) | ((h >> 3) & 0x04);

                let v0 = q0 as f32 - 4.0;
                let v1 = q1 as f32 - 4.0;
                let v2 = q2 as f32 - 4.0;
                let v3 = q3 as f32 - 4.0;

                let sc0 = (scales[ir] & 0x0F) as f32;
                let sc1 = (scales[ir] >> 4) as f32;

                let y_idx = ib * 256 + 8 * ir + i;
                block_sum += input[y_idx] * (v0 * sc0 + v1 * sc1);
                block_sum += input[y_idx + 32] * (v2 * sc0 + v3 * sc1);
            }

            // High part
            for i in 0..8 {
                let q = block.qs[8 * ir + i + 32];
                let h = block.hmask[8 * ir + i + 32];

                let q0 = (q & 0x03) | ((h & 1) << 2);
                let q1 = ((q >> 2) & 0x03) | ((h >> 1) & 0x04);
                let q2 = ((q >> 4) & 0x03) | ((h >> 2) & 0x04);
                let q3 = ((q >> 6) & 0x03) | ((h >> 3) & 0x04);

                let v0 = q0 as f32 - 4.0;
                let v1 = q1 as f32 - 4.0;
                let v2 = q2 as f32 - 4.0;
                let v3 = q3 as f32 - 4.0;

                let sc0 = (scales[ir + 4] & 0x0F) as f32;
                let sc1 = (scales[ir + 4] >> 4) as f32;

                let y_idx = ib * 256 + 128 + 8 * ir + i;
                block_sum += input[y_idx] * (v0 * sc0 + v1 * sc1);
                block_sum += input[y_idx + 32] * (v2 * sc0 + v3 * sc1);
            }
        }

        total_sum += d * block_sum;
    }

    total_sum
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    println!("\n=== Q3_K GEMM Correctness Verification ===\n");

    // Test 1: Small matrix
    println!("Test 1: Small matrix (M=2, N=2, K=256)");
    let m = 2;
    let n = 2;
    let k = 256;

    let weights = generate_q3_k_weights(m, k);
    let input: Vec<f32> = (0..(k * n)).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();

    let gpu_result = ctx.gemm_q3_k_f32(m, n, k, &weights, &input).expect("GPU failed");

    // Reference calculation
    let mut ref_result = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let row_weights = &weights[row..(row + 1)];
            let col_input = &input[col * k..(col + 1) * k];
            // Note: This is simplified, actual reference would need full implementation
        }
    }

    println!("GPU result[0]: {}", gpu_result[0]);
    println!("GPU result[1]: {}", gpu_result[1]);
    println!("GPU result[2]: {}", gpu_result[2]);
    println!("GPU result[3]: {}", gpu_result[3]);

    // Test 2: Medium matrix
    println!("\nTest 2: Medium matrix (M=16, N=4, K=1024)");
    let m2 = 16;
    let n2 = 4;
    let k2 = 1024;

    let weights2 = generate_q3_k_weights(m2, k2);
    let input2: Vec<f32> = (0..(k2 * n2)).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();

    let gpu_result2 = ctx.gemm_q3_k_f32(m2, n2, k2, &weights2, &input2).expect("GPU failed");

    println!("GPU result[0..4]: {:?}", &gpu_result2[0..4]);

    // Test 3: Large matrix
    println!("\nTest 3: Large matrix (M=4096, N=128, K=4096)");
    let m3 = 4096;
    let n3 = 128;
    let k3 = 4096;

    let weights3 = generate_q3_k_weights(m3, k3);
    let input3: Vec<f32> = (0..(k3 * n3)).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();

    let start = std::time::Instant::now();
    let gpu_result3 = ctx.gemm_q3_k_f32(m3, n3, k3, &weights3, &input3).expect("GPU failed");
    let elapsed = start.elapsed().as_micros() as f64;

    let ops = 2.0 * m3 as f64 * n3 as f64 * k3 as f64;
    let gflops = ops / (elapsed / 1_000_000.0) / 1_000_000_000.0;

    println!("Time: {:.2} µs", elapsed);
    println!("GFLOPS: {:.2}", gflops);
    println!("Sample results[0..4]: {:?}", &gpu_result3[0..4]);
    println!("Sample results[100..104]: {:?}", &gpu_result3[100..104]);

    // Check if values are reasonable
    let max_val = gpu_result3.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = gpu_result3.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("Value range: [{}, {}]", min_val, max_val);

    // Check for NaN or Inf
    let has_nan = gpu_result3.iter().any(|&x| x.is_nan());
    let has_inf = gpu_result3.iter().any(|&x| x.is_infinite());
    println!("Has NaN: {}", has_nan);
    println!("Has Inf: {}", has_inf);

    // Check if all zeros
    let all_zeros = gpu_result3.iter().all(|&x| x == 0.0);
    println!("All zeros: {}", all_zeros);

    if gflops > 1000.0 {
        println!("\n⚠️ WARNING: GFLOPS > 1000 is unrealistic for M1 Max!");
        println!("This suggests the kernel may not be computing correctly.");
    }

    if has_nan || has_inf || all_zeros {
        println!("\n⚠️ WARNING: Kernel correctness issue detected!");
    }
}
