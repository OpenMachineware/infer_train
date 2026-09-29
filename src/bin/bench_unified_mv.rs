// Unified MV benchmark: llama.cpp Metal vs Rust Metal
// Tests both in the same session for fair comparison

use std::hint::black_box;
use std::time::Instant;

use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::*;

const QK_K: usize = 256;
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
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    println!("GPU warmup complete\n");
}

fn generate_q4k_blocks(nb: usize, seed: usize) -> Vec<BlockQ4K> {
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((j + 1) % 4) as u8 + 1;
            }
            for j in 0..128 {
                qs[j] = ((i + j + seed) & 0xFF) as u8;
            }
            BlockQ4K {
                d: 15360, // FP16 1.0
                dmin: 0,
                scales,
                qs,
            }
        })
        .collect()
}

fn bench_mv_q4k(
    ctx: &MetalContext,
    m: usize,
    k: usize,
    iterations: usize,
) -> f64 {
    let nb = m * (k / QK_K);
    let weights = generate_q4k_blocks(nb, 42);
    let input = vec![1.0f32; k];

    // Warmup
    for _ in 0..3 {
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(ctx.mv_q4_k_f32(m, k, &weights, &input));
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;

    // Calculate GFLOPS: 2 * m * k FLOPs per iteration
    let flops = 2.0 * (m as f64) * (k as f64) * (iterations as f64);
    flops / (elapsed_ns / 1e9) / 1e9
}

fn main() {
    println!("=== Unified MV Benchmark: Q4_K ===\n");

    let device = Device::system_default().expect("No Metal device found");
    println!("Metal device: {}\n", device.name());

    // Run multiple rounds to see variance
    let sizes = [(4096, 256), (4096, 1024), (4096, 4096), (8192, 4096)];
    let iterations = [1000, 500, 200, 100];

    println!("Round 1: Fresh GPU (after warmup)");
    warmup_gpu(&device);
    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    for (idx, (m, k)) in sizes.iter().enumerate() {
        let iter = iterations[idx];
        let gflops = bench_mv_q4k(&ctx, *m, *k, iter);
        println!("Q4_K M={} K={}: {:.2} GFLOPS", m, k, gflops);
    }

    println!("\nRound 2: Immediate re-run (no extra warmup)");
    for (idx, (m, k)) in sizes.iter().enumerate() {
        let iter = iterations[idx];
        let gflops = bench_mv_q4k(&ctx, *m, *k, iter);
        println!("Q4_K M={} K={}: {:.2} GFLOPS", m, k, gflops);
    }

    println!("\nRound 3: After 2 second delay");
    std::thread::sleep(std::time::Duration::from_secs(2));
    for (idx, (m, k)) in sizes.iter().enumerate() {
        let iter = iterations[idx];
        let gflops = bench_mv_q4k(&ctx, *m, *k, iter);
        println!("Q4_K M={} K={}: {:.2} GFLOPS", m, k, gflops);
    }

    println!("\n=== Test Complete ===");
    println!("\nObservation:");
    println!(
        "- If Round 1 and Round 2 differ significantly, GPU frequency is unstable"
    );
    println!("- If Round 3 differs from Round 2, idle time affects frequency");
    println!("- This explains the variance we see in single-run benchmarks");
}
