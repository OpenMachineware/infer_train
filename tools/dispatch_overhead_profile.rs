// Compare single-kernel vs batched dispatch bandwidth
// Tests if dispatch overhead is the main issue

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::MetalContext;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::MTLResourceOptions;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    println!("=== Single vs Batched Dispatch Bandwidth ===\n");

    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let device = ctx.device();

    // Create test data for 6 GEMV kernels (simulating one decode token)
    let kernels = vec![
        (2048, 1024), // Q
        (1024, 1024), // K
        (1024, 1024), // V
        (3072, 1024), // Gate
        (3072, 1024), // Up
        (1024, 3072), // Down
    ];

    let num_iterations = 20;

    // Test 1: Separate command buffers (what we're measuring now)
    println!("Test 1: Separate command buffers (6 dispatches)");
    let mut total_time_separate = 0.0;

    for _ in 0..num_iterations {
        let start = std::time::Instant::now();
        for (m, k) in &kernels {
            let qk_k = 256;
            let weight_bytes = (k / qk_k) * 144 * m;
            let weight_buffer = device.new_buffer(
                weight_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let input_buffer = device.new_buffer(
                (k * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let _ = ctx.mv_q4_k_f32_gpu_weights(
                *m,
                *k,
                &weight_buffer,
                &input_buffer,
            );
        }
        total_time_separate += start.elapsed().as_nanos() as f64;
    }

    let avg_separate_us = total_time_separate / num_iterations as f64 / 1000.0;
    let total_bytes: usize = kernels
        .iter()
        .map(|(m, k)| {
            let weight_bytes = (k / 256) * 144 * m;
            weight_bytes + k * 4 + m * 4
        })
        .sum();
    let bw_separate = (total_bytes as f64) / (avg_separate_us * 1e-6) / 1e9;

    println!("  Total time: {:.2} μs", avg_separate_us);
    println!("  Bandwidth: {:.2} GB/s", bw_separate);
    println!("  Utilization: {:.1}%", bw_separate / 100.0 * 100.0);

    // Test 2: Single command buffer (like actual decode)
    println!("\nTest 2: Single command buffer (batched)");

    // Pre-create buffers
    let mut buffers = Vec::new();
    for (m, k) in &kernels {
        let qk_k = 256;
        let weight_bytes = (k / qk_k) * 144 * m;
        let weight_buffer = device.new_buffer(
            weight_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device
            .new_buffer((k * 4) as u64, MTLResourceOptions::StorageModeShared);
        let output_buffer = device
            .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
        buffers.push((weight_buffer, input_buffer, output_buffer, *m, *k));
    }

    // Warm-up
    for _ in 0..3 {
        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        for (weight, input, output, m, k) in &buffers {
            ctx.mv_encode(
                encoder,
                weight,
                input,
                output,
                *m,
                *k,
                infer_train::model::tensor::DataType::Q4_K,
            );
        }
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Measure
    let mut total_time_batched = 0.0;
    for _ in 0..num_iterations {
        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        for (weight, input, output, m, k) in &buffers {
            ctx.mv_encode(
                encoder,
                weight,
                input,
                output,
                *m,
                *k,
                infer_train::model::tensor::DataType::Q4_K,
            );
        }
        encoder.end_encoding();

        let start = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        total_time_batched += start.elapsed().as_nanos() as f64;
    }

    let avg_batched_us = total_time_batched / num_iterations as f64 / 1000.0;
    let bw_batched = (total_bytes as f64) / (avg_batched_us * 1e-6) / 1e9;

    println!("  Total time: {:.2} μs", avg_batched_us);
    println!("  Bandwidth: {:.2} GB/s", bw_batched);
    println!("  Utilization: {:.1}%", bw_batched / 100.0 * 100.0);

    println!("\n=== Analysis ===");
    println!("Speedup from batching: {:.2}x", avg_separate_us / avg_batched_us);
    println!(
        "Bandwidth improvement: {:.1}pp",
        bw_batched / 100.0 * 100.0 - bw_separate / 100.0 * 100.0
    );
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!(
        "This benchmark requires Metal support (macOS with metal feature)"
    );
}
