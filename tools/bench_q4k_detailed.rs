// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed Q4_K GEMV benchmark: our kernel vs llama.cpp kernel
//! Identifies performance differences at the kernel level

use metal::*;
use std::time::Instant;

fn create_q4_k_test_data(
    device: &DeviceRef,
    m: usize,
    k: usize,
) -> (Buffer, Buffer, Buffer, Vec<f32>) {
    let nb = k / 256;
    let weights_size = m * nb * 144; // Q4_K block size is 144 bytes
    let mut weights_data = vec![0u8; weights_size];

    // Create valid Q4_K weights
    for row in 0..m {
        for block in 0..nb {
            let offset = (row * nb + block) * 144;
            // Set d and dmin to 1.0 (FP16 0x3C00)
            let d_bytes = 0x3C00u16.to_le_bytes();
            weights_data[offset] = d_bytes[0];
            weights_data[offset + 1] = d_bytes[1];
            weights_data[offset + 2] = d_bytes[0];
            weights_data[offset + 3] = d_bytes[1];
            // Set scales to middle values
            for i in 4..16 {
                weights_data[offset + i] = 32; // Middle scale value
            }
            // Set quantized values to zeros
            for i in 16..144 {
                weights_data[offset + i] = 0;
            }
        }
    }

    let weights_buffer = device.new_buffer_with_data(
        weights_data.as_ptr() as *const std::ffi::c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create input vector
    let input_data: Vec<f32> = vec![1.0; k];
    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create output buffer
    let output_buffer = device.new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    // Compute reference output
    let mut reference = vec![0.0f32; m];
    // For our simple weights (d=1.0, scales=32, qs=0), output should be 0

    (weights_buffer, input_buffer, output_buffer, reference)
}

#[repr(C)]
struct KquantMvArgs {
    ne00: u32, // K
    ne01: u32, // M
    nb01: u64, // Byte stride between rows
}

fn main() {
    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // Load our kernel
    let our_source = include_str!("../src/ops/metal/gemv/kquant_mv.metal");
    let blocks = include_str!("../src/ops/metal/include/blocks.h");
    let our_combined = format!("#define GGML_COMMON_DECL_METAL\n{}\n{}", blocks, our_source);

    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library = device
        .new_library_with_source(&our_combined, &options)
        .expect("Failed to compile our library");

    let our_kernel = our_library
        .get_function("kernel_mul_mv_q4_K_f32", None)
        .expect("Failed to get our kernel");

    let our_pipeline = device
        .new_compute_pipeline_state_with_function(&our_kernel)
        .expect("Failed to create our pipeline");

    // Load llama.cpp kernel
    let llama_source = include_str!("../llama.cpp-0.7.0/ggml/src/ggml-metal/kernels/mul_mv.metal");
    let llama_common = include_str!("../llama.cpp-0.7.0/ggml/src/ggml-metal/kernels/common.h");
    let llama_dequant = include_str!("../llama.cpp-0.7.0/ggml/src/ggml-metal/kernels/dequantize.h");

    // Create a minimal llama.cpp kernel setup
    let llama_combined = format!(
        "{}\n{}\n{}",
        llama_common, llama_dequant, llama_source
    );

    let llama_library = device
        .new_library_with_source(&llama_combined, &options)
        .expect("Failed to compile llama.cpp library");

    let llama_kernel = llama_library
        .get_function("kernel_mul_mv_q4_K_f32", None)
        .expect("Failed to get llama.cpp kernel");

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama.cpp pipeline");

    // Test cases (decode scenario: many rows, small K)
    let test_cases = vec![
        ("Q projection", 2048, 1024),
        ("K projection", 1024, 1024),
        ("V projection", 1024, 1024),
        ("Gate projection", 3072, 1024),
        ("Up projection", 3072, 1024),
        ("Down projection", 1024, 3072),
    ];

    println!("Q4_K GEMV Kernel Comparison: Ours vs llama.cpp");
    println!("================================================\n");

    let num_warmup = 5;
    let num_iterations = 100;

    for (name, m, k) in test_cases {
        let (weights, input, output, _reference) = create_q4_k_test_data(&device, m, k);

        let args = KquantMvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01: ((k / 256) * 144) as u64, // Q4_K block size = 144
        };

        // Test our kernel
        let our_times: Vec<f64> = (0..num_iterations)
            .map(|i| {
                let cmd_buffer = queue.new_command_buffer();
                let encoder = cmd_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(&our_pipeline);
                encoder.set_buffer(0, &weights);
                encoder.set_buffer(1, &input);
                encoder.set_buffer(2, &output);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<KquantMvArgs>() as u64,
                    &args as *const KquantMvArgs as *const std::ffi::c_void,
                );

                // NSG=2, NR0=2
                let nsg = 2;
                let nr0 = 2;
                let threads_per_tg = nsg * 32;
                let tgs = (m + nsg * nr0 - 1) / (nsg * nr0);

                let thread_group_size = MTLSize {
                    width: 32,
                    height: nsg as u64,
                    depth: 1,
                };
                let thread_group_count = MTLSize {
                    width: tgs as u64,
                    height: 1,
                    depth: 1,
                };

                encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
                encoder.end_encoding();

                if i >= num_warmup {
                    let start = Instant::now();
                    cmd_buffer.commit();
                    cmd_buffer.wait_until_completed();
                    start.elapsed().as_secs_f64() * 1_000_000.0 // microseconds
                } else {
                    cmd_buffer.commit();
                    cmd_buffer.wait_until_completed();
                    0.0
                }
            })
            .filter(|&t| t > 0.0)
            .collect();

        // Test llama.cpp kernel
        let llama_times: Vec<f64> = (0..num_iterations)
            .map(|i| {
                let cmd_buffer = queue.new_command_buffer();
                let encoder = cmd_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(&llama_pipeline);
                encoder.set_buffer(0, &weights);
                encoder.set_buffer(1, &input);
                encoder.set_buffer(2, &output);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<KquantMvArgs>() as u64,
                    &args as *const KquantMvArgs as *const std::ffi::c_void,
                );

                let nsg = 2;
                let nr0 = 2;
                let threads_per_tg = nsg * 32;
                let tgs = (m + nsg * nr0 - 1) / (nsg * nr0);

                let thread_group_size = MTLSize {
                    width: 32,
                    height: nsg as u64,
                    depth: 1,
                };
                let thread_group_count = MTLSize {
                    width: tgs as u64,
                    height: 1,
                    depth: 1,
                };

                encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
                encoder.end_encoding();

                if i >= num_warmup {
                    let start = Instant::now();
                    cmd_buffer.commit();
                    cmd_buffer.wait_until_completed();
                    start.elapsed().as_secs_f64() * 1_000_000.0 // microseconds
                } else {
                    cmd_buffer.commit();
                    cmd_buffer.wait_until_completed();
                    0.0
                }
            })
            .filter(|&t| t > 0.0)
            .collect();

        // Calculate statistics
        let our_avg: f64 = our_times.iter().sum::<f64>() / our_times.len() as f64;
        let llama_avg: f64 = llama_times.iter().sum::<f64>() / llama_times.len() as f64;

        let bytes_accessed = (m * k * 4 + m * k / 16 * 144) as f64; // Input + weights
        let our_bw = bytes_accessed / our_avg / 1000.0; // GB/s
        let llama_bw = bytes_accessed / llama_avg / 1000.0; // GB/s

        let theoretical_bw = 100.0; // GB/s for M1 Max
        let our_util = our_bw / theoretical_bw * 100.0;
        let llama_util = llama_bw / theoretical_bw * 100.0;

        let speedup = our_avg / llama_avg;

        println!("Test: {} (M={}, K={})", name, m, k);
        println!("  Our kernel:    {:7.2} μs, {:6.2f} GB/s, {:5.2f}% util", our_avg, our_bw, our_util);
        println!("  llama.cpp:     {:7.2} μs, {:6.2f} GB/s, {:5.2f}% util", llama_avg, llama_bw, llama_util);
        println!("  Speedup:       {:5.2f}x ({:.1}% {})", speedup, (speedup - 1.0).abs() * 100.0, if speedup < 1.0 { "FASTER" } else { "SLOWER" });
        println!();
    }
}