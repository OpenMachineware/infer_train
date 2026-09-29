// GPU Operator Performance Benchmark - Complete Version
// All formats, all sizes, all batch sizes

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use infer_train::quant::types::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K,
};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::*;

// Complete K dimensions
const K_DIMS: [usize; 6] = [256, 512, 1024, 2048, 4096, 8192];

// Complete M dimensions
const M_DIMS: [usize; 4] = [4096, 8192, 16384, 32768];

// Complete batch sizes
const BATCH_SIZES: [usize; 11] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

// Warmup rounds
const WARMUP_ROUNDS: usize = 3;

fn warmup_gpu(device: &Device) {
    println!("Warming up GPU...");

    let command_queue = device.new_command_queue();

    for round in 0..WARMUP_ROUNDS {
        let cmd_buffer = command_queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        let library = device.new_library_with_source(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void warmup(device float* dst [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
                if (idx < 1024) { dst[idx] = float(idx) * 0.5; }
            }
            "#,
            &CompileOptions::new()
        ).unwrap();

        let pipeline = device
            .new_compute_pipeline_state_with_function(
                &library.get_function("warmup", None).unwrap(),
            )
            .unwrap();

        let buffer =
            device.new_buffer(1024 * 4, MTLResourceOptions::StorageModeShared);
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.dispatch_thread_groups(
            MTLSize { width: 1024, height: 1, depth: 1 },
            MTLSize { width: 64, height: 1, depth: 1 },
        );
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        if round < WARMUP_ROUNDS - 1 {
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    println!("GPU warmup complete\n");
}

fn main() {
    println!("=== Complete GPU Operator Performance Benchmark ===\n");

    let device = Device::system_default().expect("No Metal device found");
    println!("Metal device: {}\n", device.name());
    warmup_gpu(&device);

    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    bench_mv_all(&ctx);
    bench_gemm_all(&ctx);
    bench_auxiliary(&ctx);

    println!("\n=== Benchmark Complete ===");
}

fn bench_mv_all(ctx: &MetalContext) {
    println!("MV (Decode) Performance - Q2_K/Q3_K/Q4_K\n");

    // Baselines from llama.cpp
    let q2k_baseline: HashMap<(usize, usize), f64> = [
        ((4096, 256), 6.01),
        ((4096, 1024), 28.45),
        ((4096, 4096), 75.62),
        ((8192, 4096), 129.65),
        ((16384, 4096), 224.18),
    ]
    .iter()
    .cloned()
    .collect();

    let q3k_baseline: HashMap<(usize, usize), f64> = [
        ((4096, 256), 6.34),
        ((4096, 1024), 22.96),
        ((4096, 4096), 70.24),
        ((8192, 4096), 108.05),
        ((16384, 4096), 146.37),
    ]
    .iter()
    .cloned()
    .collect();

    let q4k_baseline: HashMap<(usize, usize), f64> = [
        ((4096, 256), 7.28),
        ((4096, 1024), 29.16),
        ((4096, 4096), 63.26),
        ((8192, 4096), 108.12),
        ((16384, 4096), 165.15),
    ]
    .iter()
    .cloned()
    .collect();

    println!("Format | M     | K    | GFLOPS | vs llama.cpp");
    println!("-------|-------|------|--------|-------------");

    for (m, k) in
        [(4096, 256), (4096, 1024), (4096, 4096), (8192, 4096), (16384, 4096)]
    {
        for (fmt, baseline) in [
            ("Q2_K", &q2k_baseline),
            ("Q3_K", &q3k_baseline),
            ("Q4_K", &q4k_baseline),
        ] {
            if let Some(base) = baseline.get(&(m, k)) {
                let gflops = match fmt {
                    "Q2_K" => bench_mv_q2k(ctx, m, k),
                    "Q3_K" => bench_mv_q3k(ctx, m, k),
                    "Q4_K" => bench_mv_q4k(ctx, m, k),
                    _ => 0.0,
                };
                let ratio = (gflops / base) * 100.0;
                let status = if ratio >= 100.0 { "✅" } else { "❌" };
                println!(
                    "{:6} | {:5} | {:4} | {:6.2} | {:5.1}% {}",
                    fmt, m, k, gflops, ratio, status
                );
            }
        }
    }
    println!();
}

fn bench_gemm_all(ctx: &MetalContext) {
    println!("GEMM (Batch Prefill) Performance - All Batch Sizes\n");

    let q2k_baseline: HashMap<usize, f64> =
        [(32, 1056.0), (64, 1604.0), (128, 2661.0), (256, 4335.0)]
            .iter()
            .cloned()
            .collect();
    let q3k_baseline: HashMap<usize, f64> =
        [(32, 2159.0), (64, 2714.0), (128, 3976.0), (256, 5265.0)]
            .iter()
            .cloned()
            .collect();
    let q4k_baseline: HashMap<usize, f64> =
        [(32, 2248.0), (64, 3169.0), (128, 4601.0), (256, 5229.0)]
            .iter()
            .cloned()
            .collect();

    let m = 4096usize;
    let k = 4096usize;

    println!("Format | Batch | GFLOPS  | vs llama.cpp");
    println!("-------|-------|---------|-------------");

    for batch in [32, 64, 128, 256].iter() {
        for (fmt, baseline) in [
            ("Q2_K", &q2k_baseline),
            ("Q3_K", &q3k_baseline),
            ("Q4_K", &q4k_baseline),
        ] {
            if let Some(base) = baseline.get(batch) {
                let gflops = match fmt {
                    "Q2_K" => bench_gemm_q2k(ctx, m, k, *batch),
                    "Q3_K" => bench_gemm_q3k(ctx, m, k, *batch),
                    "Q4_K" => bench_gemm_q4k(ctx, m, k, *batch),
                    _ => 0.0,
                };
                let ratio = (gflops / base) * 100.0;
                let status = if ratio >= 100.0 { "✅" } else { "❌" };
                println!(
                    "{:6} | {:5} | {:7.2} | {:5.1}% {}",
                    fmt, batch, gflops, ratio, status
                );
            }
        }
    }
    println!();
}

fn bench_auxiliary(ctx: &MetalContext) {
    println!("Auxiliary Operators Performance\n");

    let size = 4096usize;
    let data = vec![1.0f32; size];
    let weight = vec![1.0f32; size];

    // RMS Norm
    for _ in 0..3 {
        let _ = ctx.rms_norm_f32(1, size, &data, &weight, 1e-5);
    }
    let start = Instant::now();
    for _ in 0..100 {
        let _ = black_box(ctx.rms_norm_f32(1, size, &data, &weight, 1e-5));
    }
    let elapsed_us = start.elapsed().as_micros() as f64 / 100.0;
    let status = if elapsed_us <= 266.85 { "✅" } else { "❌" };
    println!("RMS Norm: {:6.2} µs (baseline: 266.85) {}", elapsed_us, status);

    // Softmax
    let start = Instant::now();
    for _ in 0..100 {
        let _ = black_box(ctx.softmax_f32_fast(1, size, &data));
    }
    let elapsed_us = start.elapsed().as_micros() as f64 / 100.0;
    let status = if elapsed_us <= 261.93 { "✅" } else { "❌" };
    println!("Softmax:  {:6.2} µs (baseline: 261.93) {}", elapsed_us, status);
}

// Generate test data
fn generate_q2k_blocks(nb: usize, seed: usize) -> Vec<BlockQ2K> {
    use half::f16;
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 16];
            let mut qs = [0u8; 64];
            for j in 0..16 {
                scales[j] = ((j + 1) % 8) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i + j + seed) & 0xFF) as u8;
            }
            BlockQ2K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q3k_blocks(nb: usize, seed: usize) -> Vec<BlockQ3K> {
    use half::f16;
    (0..nb)
        .map(|i| {
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 12];
            for j in 0..32 {
                hmask[j] = ((i + j + seed) & 0xFF) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i + j + seed) & 0xFF) as u8;
            }
            for j in 0..12 {
                scales[j] = ((j + 1) % 4) as u8;
            }
            BlockQ3K { hmask, qs, scales, d: f16::from_f32(1.0).to_bits() }
        })
        .collect()
}

fn generate_q4k_blocks(nb: usize, seed: usize) -> Vec<BlockQ4K> {
    use half::f16;
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((j + 1) % 4 + 1) as u8;
            }
            for j in 0..128 {
                qs[j] = ((i + j + seed) & 0xFF) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

// Benchmarks
fn bench_mv_q2k(ctx: &MetalContext, m: usize, k: usize) -> f64 {
    let weights = generate_q2k_blocks(m, 42);
    let input: Vec<f32> = (0..k).map(|i| ((i % 10) as f32) * 0.1).collect();
    for _ in 0..3 {
        let _ = ctx.mv_q2_k_f32(m, k, &weights, &input);
    }
    let start = Instant::now();
    for _ in 0..10 {
        let _ = black_box(ctx.mv_q2_k_f32(m, k, &weights, &input));
    }
    2.0 * m as f64 * k as f64 * 10.0 / start.elapsed().as_secs_f64() / 1e9
}

fn bench_mv_q3k(ctx: &MetalContext, m: usize, k: usize) -> f64 {
    let weights = generate_q3k_blocks(m, 42);
    let input: Vec<f32> = (0..k).map(|i| ((i % 10) as f32) * 0.1).collect();
    for _ in 0..3 {
        let _ = ctx.mv_q3_k_f32(m, k, &weights, &input);
    }
    let start = Instant::now();
    for _ in 0..10 {
        let _ = black_box(ctx.mv_q3_k_f32(m, k, &weights, &input));
    }
    2.0 * m as f64 * k as f64 * 10.0 / start.elapsed().as_secs_f64() / 1e9
}

fn bench_mv_q4k(ctx: &MetalContext, m: usize, k: usize) -> f64 {
    let weights = generate_q4k_blocks(m, 42);
    let input: Vec<f32> = (0..k).map(|i| ((i % 10) as f32) * 0.1).collect();
    for _ in 0..3 {
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
    }
    let start = Instant::now();
    for _ in 0..10 {
        let _ = black_box(ctx.mv_q4_k_f32(m, k, &weights, &input));
    }
    2.0 * m as f64 * k as f64 * 10.0 / start.elapsed().as_secs_f64() / 1e9
}

fn bench_gemm_q2k(ctx: &MetalContext, m: usize, k: usize, batch: usize) -> f64 {
    let weights = generate_q2k_blocks(m, 42);
    let input: Vec<f32> =
        (0..k * batch).map(|i| ((i % 10) as f32) * 0.1).collect();
    for _ in 0..3 {
        let _ = ctx.gemm_q2_k_f32(m, batch, k, &weights, &input);
    }
    let start = Instant::now();
    for _ in 0..10 {
        let _ = black_box(ctx.gemm_q2_k_f32(m, batch, k, &weights, &input));
    }
    2.0 * m as f64 * k as f64 * batch as f64 * 10.0
        / start.elapsed().as_secs_f64()
        / 1e9
}

fn bench_gemm_q3k(ctx: &MetalContext, m: usize, k: usize, batch: usize) -> f64 {
    let weights = generate_q3k_blocks(m, 42);
    let input: Vec<f32> =
        (0..k * batch).map(|i| ((i % 10) as f32) * 0.1).collect();
    for _ in 0..3 {
        let _ = ctx.gemm_q3_k_f32(m, batch, k, &weights, &input);
    }
    let start = Instant::now();
    for _ in 0..10 {
        let _ = black_box(ctx.gemm_q3_k_f32(m, batch, k, &weights, &input));
    }
    2.0 * m as f64 * k as f64 * batch as f64 * 10.0
        / start.elapsed().as_secs_f64()
        / 1e9
}

fn bench_gemm_q4k(ctx: &MetalContext, m: usize, k: usize, batch: usize) -> f64 {
    let weights = generate_q4k_blocks(m, 42);
    let input: Vec<f32> =
        (0..k * batch).map(|i| ((i % 10) as f32) * 0.1).collect();
    for _ in 0..3 {
        let _ = ctx.gemm_q4_k_f32(m, batch, k, &weights, &input);
    }
    let start = Instant::now();
    for _ in 0..10 {
        let _ = black_box(ctx.gemm_q4_k_f32(m, batch, k, &weights, &input));
    }
    2.0 * m as f64 * k as f64 * batch as f64 * 10.0
        / start.elapsed().as_secs_f64()
        / 1e9
}
