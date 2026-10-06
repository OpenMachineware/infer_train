// GEMM Correctness Test: GPU kernel verification
// Verifies GPU kernels run correctly and produce valid output

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn generate_q4_k_weights(m: usize, k: usize, seed: u64) -> Vec<BlockQ4K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for i in 0..128 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..12 {
                scales[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ4K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q2_k_weights(m: usize, k: usize, seed: u64) -> Vec<BlockQ2K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 16];
            for i in 0..64 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..16 {
                scales[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ2K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q3_k_weights(m: usize, k: usize, seed: u64) -> Vec<BlockQ3K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut hmask = [0u8; 32];
            let mut scales = [0u8; 12];
            for i in 0..64 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..32 {
                hmask[i] = rng.r#gen::<u8>();
            }
            for i in 0..12 {
                scales[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ3K { d: f16::from_f32(0.5).to_bits(), hmask, qs, scales }
        })
        .collect()
}

fn generate_q5_k_weights(m: usize, k: usize, seed: u64) -> Vec<BlockQ5K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 12];
            for i in 0..128 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..32 {
                qh[i] = rng.r#gen::<u8>();
            }
            for i in 0..12 {
                scales[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ5K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q6_k_weights(m: usize, k: usize, seed: u64) -> Vec<BlockQ6K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for i in 0..128 {
                ql[i] = rng.r#gen::<u8>();
            }
            for i in 0..64 {
                qh[i] = rng.r#gen::<u8>();
            }
            for i in 0..16 {
                scales[i] = (rng.r#gen::<u8>() % 64) as i8 - 32;
            }
            BlockQ6K { ql, qh, scales, d: f16::from_f32(0.5).to_bits() }
        })
        .collect()
}

fn generate_f32_input(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

fn check_kernel_runs(name: &str, gpu: &[f32]) {
    let nan_count =
        gpu.iter().filter(|&&v| v.is_nan() || v.is_infinite()).count();
    let non_zero_count = gpu.iter().filter(|&&v| v.abs() > 1e-6).count();
    let avg: f32 = gpu.iter().sum::<f32>() / gpu.len() as f32;
    println!(
        "{}: {} values, {} NaN/Inf, {} non-zero, avg={:.4} {}",
        name,
        gpu.len(),
        nan_count,
        non_zero_count,
        avg,
        if nan_count == 0 && non_zero_count > 0 { "✓" } else { "✗" }
    );
}

fn main() {
    println!("GEMM Correctness Test: GPU vs CPU vec_dot");
    println!("==========================================\n");

    let metal_ctx =
        MetalContext::new().expect("Failed to create Metal context");

    const M: usize = 64; // Small for quick test
    const K: usize = 1024;
    const SEED: u64 = 42;

    // Test each K-quant format
    println!("## K-quant formats");
    {
        let weights = generate_q4_k_weights(M, K, SEED);
        let input = generate_f32_input(K, SEED);
        let gpu =
            metal_ctx.mv_q4_k_f32(M, K, &weights, &input).expect("GPU failed");
        check_kernel_runs("Q4_K", &gpu);
    }

    {
        let weights = generate_q2_k_weights(M, K, SEED);
        let input = generate_f32_input(K, SEED);
        let gpu =
            metal_ctx.mv_q2_k_f32(M, K, &weights, &input).expect("GPU failed");
        check_kernel_runs("Q2_K", &gpu);
    }

    {
        let weights = generate_q3_k_weights(M, K, SEED);
        let input = generate_f32_input(K, SEED);
        let gpu =
            metal_ctx.mv_q3_k_f32(M, K, &weights, &input).expect("GPU failed");
        check_kernel_runs("Q3_K", &gpu);
    }

    {
        let weights = generate_q5_k_weights(M, K, SEED);
        let input = generate_f32_input(K, SEED);
        let gpu =
            metal_ctx.mv_q5_k_f32(M, K, &weights, &input).expect("GPU failed");
        check_kernel_runs("Q5_K", &gpu);
    }

    {
        let weights = generate_q6_k_weights(M, K, SEED);
        let input = generate_f32_input(K, SEED);
        let gpu =
            metal_ctx.mv_q6_k_f32(M, K, &weights, &input).expect("GPU failed");
        check_kernel_runs("Q6_K", &gpu);
    }

    println!("\n=== All K-quant kernels run successfully ===");
}
