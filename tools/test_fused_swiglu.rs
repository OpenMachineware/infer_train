// Test FusedSwiGLU kernel correctness
// Compares fused kernel output with separate SiLU + Mul operations

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::{Device, MTLResourceOptions};

fn main() {
    println!("=== FusedSwiGLU Kernel Test ===\n");

    // Create GPU context
    let gpu_ctx = MetalContext::new().expect("Failed to create Metal context");
    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    // Test sizes
    let test_sizes = vec![64, 256, 1024, 4096, 16384, 65536];

    for size in test_sizes {
        println!("Testing size: {}", size);

        // Generate test data
        let mut gate_data: Vec<f32> =
            (0..size).map(|i| (i as f32 * 0.1 - 5.0).sin()).collect();
        let mut up_data: Vec<f32> =
            (0..size).map(|i| (i as f32 * 0.15 - 3.0).cos()).collect();

        // Reference output: SiLU(gate) * up
        let mut reference: Vec<f32> = Vec::with_capacity(size);
        for i in 0..size {
            let g = gate_data[i];
            let u = up_data[i];
            let silu_g = g / (1.0 + (-g).exp());
            reference.push(silu_g * u);
        }

        // Upload to GPU
        let gate_buffer = device.new_buffer_with_data(
            gate_data.as_ptr() as *const std::ffi::c_void,
            (size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let up_buffer = device.new_buffer_with_data(
            up_data.as_ptr() as *const std::ffi::c_void,
            (size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = device.new_buffer(
            (size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Method 1: Separate SiLU + Mul
        let silu_buffer = device.new_buffer(
            (size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cb1 = queue.new_command_buffer();
        let encoder1 = cb1.new_compute_command_encoder();

        gpu_ctx.silu_encode(encoder1, &gate_buffer, &silu_buffer, size);
        gpu_ctx.mul_encode(
            encoder1,
            &silu_buffer,
            &up_buffer,
            &output_buffer,
            size,
        );

        encoder1.end_encoding();
        cb1.commit();
        cb1.wait_until_completed();

        // Read separate output
        let mut separate_output: Vec<f32> = vec![0.0; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                separate_output.as_mut_ptr(),
                size,
            );
        }

        // Method 2: Fused SwiGLU
        let fused_buffer = device.new_buffer(
            (size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cb2 = queue.new_command_buffer();
        let encoder2 = cb2.new_compute_command_encoder();

        gpu_ctx.fused_swiglu_encode(
            encoder2,
            &gate_buffer,
            &up_buffer,
            &fused_buffer,
            size,
        );

        encoder2.end_encoding();
        cb2.commit();
        cb2.wait_until_completed();

        // Read fused output
        let mut fused_output: Vec<f32> = vec![0.0; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                fused_buffer.contents() as *const f32,
                fused_output.as_mut_ptr(),
                size,
            );
        }

        // Compare results
        let mut max_rel_error = 0.0f32;
        let mut max_abs_error = 0.0f32;
        let mut nan_count = 0;

        for i in 0..size {
            if fused_output[i].is_nan() {
                nan_count += 1;
            } else {
                let abs_error = (fused_output[i] - reference[i]).abs();
                let rel_error = if reference[i].abs() > 1e-6 {
                    abs_error / reference[i].abs()
                } else {
                    abs_error
                };
                max_abs_error = max_abs_error.max(abs_error);
                max_rel_error = max_rel_error.max(rel_error);
            }
        }

        println!("  Max abs error: {:.10}", max_abs_error);
        println!("  Max rel error: {:.10}%", max_rel_error * 100.0);
        println!("  NaN count: {}", nan_count);

        // Check correctness
        let separate_matches = separate_output
            .iter()
            .zip(reference.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        let fused_matches = fused_output
            .iter()
            .zip(reference.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5);

        if separate_matches && fused_matches {
            println!("  ✓ PASSED");
        } else {
            println!("  ✗ FAILED");
            if !separate_matches {
                println!("    Separate SiLU+Mul output mismatch");
            }
            if !fused_matches {
                println!("    Fused SwiGLU output mismatch");
            }
        }
        println!();
    }

    println!("=== Test Complete ===");
}
