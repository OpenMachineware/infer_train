// Benchmark the templated GEMM kernel from llama.cpp
// Compare performance with our hand-written kernel

use infer_train::quant::types::BlockQ4K;
use metal::{Device, MTLSize};
use std::time::Instant;

fn main() {
    println!("=== Benchmarking Templated GEMM Kernel ===\n");

    // Test dimensions matching llama.cpp: M=4096, K=4096, N=512
    let m = 4096;
    let k = 4096;
    let n = 512;

    println!("Matrix dimensions: A(M={}, K={}) x B(K={}, N={}) -> C(M={}, N={})",
             m, k, k, n, m, n);
    println!("Operations: {:.2} GFLOPS\n", (2.0 * m as f64 * k as f64 * n as f64) / 1e9);

    // Create real quantized data
    let num_blocks = k / 256;
    let mut weights_data = vec![0u8; m * num_blocks * std::mem::size_of::<BlockQ4K>()];
    let weights_ptr = weights_data.as_mut_ptr() as *mut BlockQ4K;

    // Initialize with simple pattern (d=1.0, all scales=1, all qs=0x88)
    unsafe {
        for i in 0..m * num_blocks {
            let block = weights_ptr.add(i);
            (*block).d = 0x3C00; // FP16 1.0
            (*block).dmin = 0x0000; // FP16 0.0
            // Set scales to represent value 8
            for j in 0..12 {
                (*block).scales[j] = 8; // scale=8, min=0
            }
            // Set all quants to 0x88 (value 8 in both nibbles)
            for j in 0..128 {
                (*block).qs[j] = 0x88;
            }
        }
    }
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr, m * num_blocks) };

    // Create FP16 input (B matrix)
    let input: Vec<u16> = (0..(k * n)).map(|i| {
        // Simple pattern: value = (i % 10) as FP16
        let val = (i % 10) as f32;
        half::f16::from_f32(val).to_bits()
    }).collect();

    // Compute CPU reference for first few outputs
    println!("Computing CPU reference for verification...");
    let cpu_start = Instant::now();
    let cpu_output = compute_cpu_reference(&weights, &input, m, k, n, num_blocks);
    let cpu_time = cpu_start.elapsed();
    println!("CPU reference computed in {:?}", cpu_time);

    // Test templated kernel
    println!("\nTesting templated kernel from llama.cpp...");
    let device = Device::system_default().expect("No Metal device found");
    let shader_source = include_str!("../../shaders/gemm_llama_cpp.metal");
    let compile_options = metal::CompileOptions::new();
    let library = device.new_library_with_source(shader_source, &compile_options)
        .expect("Failed to compile shader");
    let kernel = library.get_function("kernel_mul_mm_q4_K_f32", None)
        .expect("Failed to get kernel");
    let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    // Run templated kernel
    let (templated_output, templated_time) = run_gemm_on_gpu(
        &device, &pipeline, &weights, &input, m, k, n
    );

    println!("Templated kernel time: {:?}", templated_time);
    println!("Templated GFLOPS: {:.2}", gflops(m, k, n, templated_time));

    // Compare with CPU reference
    let max_diff = compare_outputs(&cpu_output, &templated_output);
    println!("\nMax difference vs CPU: {:.6}", max_diff);

    // Print first few outputs
    println!("\nFirst 5 outputs:");
    for i in 0..5 {
        println!("  CPU: {:.4}, GPU: {:.4}, diff: {:.4}",
                 cpu_output[i], templated_output[i],
                 (cpu_output[i] - templated_output[i]).abs());
    }

    // Now test our existing hand-written kernel for comparison
    println!("\n=== Comparison with existing kernel ===");
    test_existing_kernel(&weights, &input, m, k, n, num_blocks);
}

fn compute_cpu_reference(
    weights: &[BlockQ4K],
    input: &[u16],
    m: usize,
    k: usize,
    n: usize,
    num_blocks: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];

    // Simple CPU implementation
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for block_idx in 0..num_blocks {
                let w_block = &weights[row * num_blocks + block_idx];

                // Dequantize and compute dot product for this block
                let scale = w_block.d as f32 / 256.0; // Simple scale
                for i in 0..128 {
                    let lo = (w_block.qs[i] & 0x0F) as f32;
                    let hi = (w_block.qs[i] >> 4) as f32;
                    let val1 = scale * lo;
                    let val2 = scale * hi;

                    let idx1 = block_idx * 256 + 2 * i;
                    let idx2 = block_idx * 256 + 2 * i + 1;

                    if idx1 < k {
                        let input_val = half::f16::from_bits(input[idx1 * n + col]).to_f32();
                        sum += val1 * input_val;
                    }
                    if idx2 < k {
                        let input_val = half::f16::from_bits(input[idx2 * n + col]).to_f32();
                        sum += val2 * input_val;
                    }
                }
            }
            output[row * n + col] = sum;
        }
    }

    output
}

fn run_gemm_on_gpu(
    device: &Device,
    pipeline: &metal::ComputePipelineState,
    weights: &[BlockQ4K],
    input: &[u16],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, std::time::Duration) {
    let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
    let input_size = input.len() * 2;
    let output_size = m * n * 4;

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        weights_size as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        input_size as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeManaged);

    // Create argument buffer
    #[repr(C)]
    struct Args {
        ne00: i32, ne01: i32, ne02: i32, ne03: i32,
        nb00: u64, nb01: u64, nb02: u64, nb03: u64,
        ne10: i32, ne11: i32, ne12: i32, ne13: i32,
        nb10: u64, nb11: u64, nb12: u64, nb13: u64,
        ne0: i32, ne1: i32, ne2: i32, ne3: i32,
        nb0: u64, nb1: u64, nb2: u64, nb3: u64,
    }

    let args = Args {
        ne00: k as i32, ne01: m as i32, ne02: 1, ne03: 1,
        nb00: 144, nb01: (k / 256 * 144) as u64, nb02: 0, nb03: 0,
        ne10: k as i32, ne11: n as i32, ne12: 1, ne13: 1,
        nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
        ne0: m as i32, ne1: n as i32, ne2: 1, ne3: 1,
        nb0: 4, nb1: (m * 4) as u64, nb2: 0, nb3: 0,
    };

    let args_buffer = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&args_buffer), 0);
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_threadgroup_memory_length(0, 8192);

    let grid_size = MTLSize::new(
        ((n + 31) / 32) as u64,
        ((m + 63) / 64) as u64,
        1,
    );
    let threadgroup_size = MTLSize::new(128, 1, 1);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    let start = Instant::now();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let duration = start.elapsed();

    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() };

    (output, duration)
}

fn gflops(m: usize, k: usize, n: usize, duration: std::time::Duration) -> f64 {
    let ops = 2.0 * m as f64 * k as f64 * n as f64;
    let secs = duration.as_secs_f64();
    ops / secs / 1e9
}

fn compare_outputs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |max, x| if x > max { x } else { max })
}

fn test_existing_kernel(
    weights: &[BlockQ4K],
    input: &[u16],
    m: usize,
    k: usize,
    n: usize,
    num_blocks: usize,
) {
    // This would test our existing hand-written kernel
    // For now, just print that we would test it
    println!("Existing kernel test not implemented yet");
    println!("Would compare templated vs hand-written performance");
}
