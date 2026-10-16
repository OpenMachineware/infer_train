// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Jia Liu

//! Test for batched GPU forward pass

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::MTLSize;

    // Create Metal context
    let gpu_ctx = MetalContext::new().expect("Failed to create Metal context");
    let device = gpu_ctx.device();

    // Test parameters
    let m = 1024; // Output dimension
    let k = 4096; // Input dimension
    let num_ops = 10; // Number of operations to batch

    println!("=== Testing batched GEMV encoding ===");
    println!("M={}, K={}, {} operations", m, k, num_ops);

    // Create test buffers
    let input_buffer = device.new_buffer(
        (k * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Initialize input with random data
    let input_ptr = input_buffer.contents() as *mut f32;
    for i in 0..k {
        unsafe {
            *input_ptr.add(i) = (i as f32 % 10.0) - 5.0;
        }
    }

    // Create Q4_0 weight buffer
    use infer_train::quant::types::BlockQ4_0;
    let nb = k / 32;
    let num_blocks = m * nb;
    let weights_buffer = device.new_buffer(
        (num_blocks * std::mem::size_of::<BlockQ4_0>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Initialize weights
    let weights_ptr = weights_buffer.contents() as *mut BlockQ4_0;
    for i in 0..num_blocks {
        unsafe {
            let block = &mut *weights_ptr.add(i);
            block.d = 0x3C00; // FP16 1.0
            block.qs = [0u8; 16];
        }
    }

    // Test 1: Individual dispatches
    let queue = gpu_ctx.queue();
    let mut individual_times = Vec::new();

    println!("\nTesting {} individual dispatches", num_ops);
    for _ in 0..num_ops {
        let start = std::time::Instant::now();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        gpu_ctx.mv_q4_0_encode(
            encoder,
            &weights_buffer,
            &input_buffer,
            &output_buffer,
            m,
            k,
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        individual_times.push(start.elapsed());
    }

    let avg_individual =
        individual_times.iter().map(|t| t.as_micros() as f64).sum::<f64>()
            / individual_times.len() as f64;
    println!("  Average: {:.2} us", avg_individual);
    println!("  Per operation: {:.2} us", avg_individual);

    // Test 2: Batched dispatches
    println!("\nTesting {} batched dispatches in one command buffer", num_ops);
    let start = std::time::Instant::now();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    for _ in 0..num_ops {
        gpu_ctx.mv_q4_0_encode(
            encoder,
            &weights_buffer,
            &input_buffer,
            &output_buffer,
            m,
            k,
        );
    }

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let batched_time = start.elapsed();

    let avg_batched = batched_time.as_micros() as f64 / num_ops as f64;
    println!("  Total: {:.2} us", batched_time.as_micros() as f64);
    println!("  Per operation: {:.2} us", avg_batched);
    println!("  Speedup: {:.2}x", avg_individual / avg_batched);

    // Test 3: Test mv_encode with different formats
    println!("\n=== Testing mv_encode with different formats ===");

    // Test Q4_K format
    use infer_train::quant::types::BlockQ4K;
    let nb_k = k / 256;
    let num_blocks_k = m * nb_k;
    let weights_buffer_k = device.new_buffer(
        (num_blocks_k * std::mem::size_of::<BlockQ4K>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let start = std::time::Instant::now();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let success = gpu_ctx.mv_encode(
        encoder,
        &weights_buffer_k,
        &input_buffer,
        &output_buffer,
        m,
        k,
        infer_train::model::tensor::DataType::Q4_K,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if success {
        println!(
            "  Q4_K encode: SUCCESS ({:.2} us)",
            start.elapsed().as_micros() as f64
        );
    } else {
        println!("  Q4_K encode: FAILED");
    }

    println!("\nBatched encoding test complete!");
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support on aarch64");
}
