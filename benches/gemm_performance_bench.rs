// GEMM Performance Benchmark with Alternating Tests
// Compare self-written mul_mm_standalone.metal vs llama.cpp's mul_mm_template.metallib

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

fn main() {
    println!("=== GEMM Performance Benchmark ===");
    println!("Testing with alternating A-B-A-B methodology\n");

    // Setup Metal
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Load libraries
    let llama_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_template.metallib")
        .expect("Failed to load llama.cpp library");

    let mul_mm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let compile_opts = CompileOptions::new();
    let self_lib = device
        .new_library_with_source(mul_mm_source, &compile_opts)
        .expect("Failed to compile self-written shader");

    println!("✓ Libraries loaded\n");

    // Test dimensions - typical LLM prompt processing
    // Large N (batch size) for prompt processing
    let test_configs = vec![
        ("Batch 64", 512, 4096, 64),
        ("Batch 256", 512, 4096, 256),
        ("Batch 1024", 512, 4096, 1024),
        ("Batch 4096", 512, 4096, 4096),
        ("Batch 8192", 512, 4096, 8192),
    ];

    // Test IQ formats
    let formats = vec![
        ("iq1_s", 256, 34),
        ("iq1_m", 256, 42),
        ("iq2_xxs", 256, 34),
        ("iq2_xs", 256, 42),
        ("iq2_s", 256, 50),
        ("iq3_xxs", 256, 98),
        ("iq3_s", 256, 106),
        ("iq4_nl", 32, 18),
        ("iq4_xs", 256, 146),
    ];

    for (config_name, m, k, n) in &test_configs {
        println!("=== {} (M={}, K={}, N={}) ===", config_name, m, k, n);

        for (format_name, block_size, block_bytes) in &formats {
            let result = benchmark_format(
                &device,
                &queue,
                &self_lib,
                &llama_lib,
                format_name,
                *m,
                *k,
                *n,
                *block_size,
                *block_bytes,
            );

            match result {
                Ok((self_time, llama_time, ratio)) => {
                    let status = if ratio < 0.98 {
                        format!("FASTER ({:.1}% faster)", (1.0 - ratio) * 100.0)
                    } else if ratio > 1.02 {
                        format!("SLOWER ({:.1}% slower)", (ratio - 1.0) * 100.0)
                    } else {
                        "TIED".to_string()
                    };
                    println!(
                        "  {}: self={:.2}ms, llama={:.2}ms, ratio={:.3} - {}",
                        format_name, self_time, llama_time, ratio, status
                    );
                }
                Err(e) => {
                    println!("  {}: ERROR - {}", format_name, e);
                }
            }
        }
        println!();
    }
}

fn benchmark_format(
    device: &Device,
    queue: &CommandQueue,
    self_lib: &Library,
    llama_lib: &Library,
    format_name: &str,
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    block_bytes: usize,
) -> Result<(f64, f64, f64), String> {
    let num_blocks = k / block_size;

    // Create test data
    let mut weights = Vec::with_capacity(m * num_blocks * block_bytes);
    let f16_2_0: [u8; 2] = [0x00, 0x40];

    // All IQ formats use simple layout with d at start
    for row in 0..m {
        for block_idx in 0..num_blocks {
            // d (f16 scale) at offset 0
            weights.push(f16_2_0[0]);
            weights.push(f16_2_0[1]);
            // Quantized data at offset 2
            for i in 0..(block_bytes - 2) {
                weights.push(((row + block_idx + i) % 256) as u8);
            }
        }
    }

    let input_f32: Vec<f32> = vec![1.0f32; n * k];
    let output_count = m * n;
    let output_bytes = output_count * 4;

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f32.as_ptr() as *const c_void,
        (input_f32.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_self = device
        .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
    let output_llama = device
        .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);

    // Arguments
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MulMMArgs {
        ne00: i32,
        ne02: i32,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        r2: i16,
        r3: i16,
    }

    let args = MulMMArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: (num_blocks * block_bytes) as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    // Function constants
    let cv = FunctionConstantValues::new();
    let bc_inp: bool = false;
    let bc_out: bool = false;
    let ne12: i16 = 1;
    let ne13: i16 = 1;
    let r2: i16 = 1;
    let r3: i16 = 1;
    unsafe {
        cv.set_constant_value_at_index(
            &bc_inp as *const _ as *const c_void,
            MTLDataType::Bool,
            700,
        );
        cv.set_constant_value_at_index(
            &bc_out as *const _ as *const c_void,
            MTLDataType::Bool,
            701,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            MTLDataType::Short,
            702,
        );
        cv.set_constant_value_at_index(
            &ne13 as *const _ as *const c_void,
            MTLDataType::Short,
            703,
        );
        cv.set_constant_value_at_index(
            &r2 as *const _ as *const c_void,
            MTLDataType::Short,
            704,
        );
        cv.set_constant_value_at_index(
            &r3 as *const _ as *const c_void,
            MTLDataType::Short,
            705,
        );
    }

    let kernel_name = format!("kernel_mul_mm_{}_f32", format_name);

    // Create pipelines
    let self_function = self_lib
        .get_function(&kernel_name, Some(cv.clone()))
        .map_err(|e| format!("Self kernel not found: {}", e))?;
    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_function)
        .map_err(|e| format!("Failed to create self pipeline: {}", e))?;

    let llama_function = llama_lib
        .get_function(&kernel_name, Some(cv))
        .map_err(|e| format!("Llama kernel not found: {}", e))?;
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_function)
        .map_err(|e| format!("Failed to create llama pipeline: {}", e))?;

    // Dispatch parameters
    let threadgroup_size = MTLSize::new(64, 2, 1);
    let threadgroups =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);

    // Warmup
    for _ in 0..5 {
        run_kernel(
            &queue,
            &self_pipeline,
            &args,
            &weights_buffer,
            &input_buffer,
            &output_self,
            threadgroup_size,
            threadgroups,
        )?;
        run_kernel(
            &queue,
            &llama_pipeline,
            &args,
            &weights_buffer,
            &input_buffer,
            &output_llama,
            threadgroup_size,
            threadgroups,
        )?;
    }

    // Alternating benchmark: A-B-A-B...
    let num_rounds = 100;
    let mut self_times = Vec::with_capacity(num_rounds);
    let mut llama_times = Vec::with_capacity(num_rounds);

    for _ in 0..num_rounds {
        let t0 = Instant::now();
        run_kernel(
            &queue,
            &self_pipeline,
            &args,
            &weights_buffer,
            &input_buffer,
            &output_self,
            threadgroup_size,
            threadgroups,
        )?;
        let self_time = t0.elapsed().as_nanos() as f64 / 1_000_000.0;

        let t1 = Instant::now();
        run_kernel(
            &queue,
            &llama_pipeline,
            &args,
            &weights_buffer,
            &input_buffer,
            &output_llama,
            threadgroup_size,
            threadgroups,
        )?;
        let llama_time = t1.elapsed().as_nanos() as f64 / 1_000_000.0;

        self_times.push(self_time);
        llama_times.push(llama_time);
    }

    // Calculate average
    let avg_self: f64 = self_times.iter().sum::<f64>() / num_rounds as f64;
    let avg_llama: f64 = llama_times.iter().sum::<f64>() / num_rounds as f64;
    let ratio = avg_self / avg_llama;

    Ok((avg_self, avg_llama, ratio))
}

fn run_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args: &impl Copy,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    threadgroup_size: MTLSize,
    threadgroups: MTLSize,
) -> Result<(), String> {
    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(
        0,
        std::mem::size_of_val(args) as u64,
        args as *const _ as *const c_void,
    );
    encoder.set_buffer(1, Some(weights), 0);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_threadgroup_memory_length(0, 8192);
    encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    encoder.end_encoding();

    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    Ok(())
}
