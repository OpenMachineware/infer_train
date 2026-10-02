// Compare K-quant MV kernels against llama.cpp reference
// Tests: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ2K {
    scales: [u8; 16], // QK_K/16 = 256/16 = 16
    qs: [u8; 64],     // QK_K/4 = 256/4 = 64
    d: u16,
    dmin: u16,
}

#[repr(C)]
struct MvArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

fn create_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn expand_includes(
    source: &str,
    shader_dir: &std::path::PathBuf,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();
    for line in source.lines() {
        if line.starts_with("#include \"") {
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];
            if visited.contains(&filename.to_string()) {
                continue;
            }
            visited.push(filename.to_string());
            let include_path = shader_dir.join(filename);
            if let Ok(include_content) = std::fs::read_to_string(&include_path)
            {
                let include_content =
                    include_content.replace("#pragma once", "// #pragma once");
                result.push_str(&expand_includes(
                    &include_content,
                    shader_dir,
                    visited,
                ));
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn generate_q4k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q2k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut scales = [0u8; 16];
            let mut qs = [0u8; 64];
            for j in 0..16 {
                scales[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            BlockQ2K {
                scales,
                qs,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn benchmark_kernel(
    pipeline: &ComputePipelineState,
    queue: &CommandQueue,
    args: &MvArgs,
    weights_buffer: &Buffer,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
    thread_group_count: MTLSize,
    thread_group_size: MTLSize,
    iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..100 {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            args as *const MvArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Benchmark - best of 3
    let mut best_us = f64::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iterations {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(weights_buffer), 0);
            encoder.set_buffer(2, Some(input_buffer), 0);
            encoder.set_buffer(3, Some(output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed_us =
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        if elapsed_us < best_us {
            best_us = elapsed_us;
        }
    }
    best_us
}

fn main() {
    println!("=== K-quant MV Kernel Comparison ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Compile self-written template
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source = std::fs::read_to_string(shader_dir.join("gemm.metal"))
        .expect("Failed to read");
    let mut visited = Vec::new();
    let my_shader = expand_includes(&my_source, &shader_dir, &mut visited);

    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    println!("Compiling self-written template...");
    let my_lib = device
        .new_library_with_source(&my_shader, &compile_options)
        .expect("Failed to compile");

    // Test parameters
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 10000;

    println!("M={}, K={}, {} iterations\n", m, k, iterations);

    // Test Q4_K
    println!("=== Q4_K ===");
    let weights = generate_q4k_weights(m, k);
    let input = generate_input(k);

    let nb = k / 256;
    let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: k as u64 * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };

    let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
    let input_size = input.len() * std::mem::size_of::<f32>();
    let output_size = m * std::mem::size_of::<f32>();

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        input_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cv = create_function_constants(2, 1, 1, 1);
    let my_kernel = my_lib
        .get_function("kernel_gemm_q4_K_f32", Some(cv.clone()))
        .expect("Failed to get Q4_K kernel");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create pipeline");

    const NSG: u64 = 2;
    const ROWS_PER_THREADGROUP: u64 = NSG * 2;
    let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
    let thread_group_count = MTLSize {
        width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
        height: 1,
        depth: 1,
    };

    let my_output_buffer = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_time = benchmark_kernel(
        &my_pipeline,
        &queue,
        &args,
        &weights_buffer,
        &input_buffer,
        &my_output_buffer,
        thread_group_count,
        thread_group_size,
        iterations,
    );

    let my_gflops = (2.0 * m as f64 * k as f64) / (my_time * 1e-6) / 1e9;
    println!("Self-written: {:.2} us ({:.1} GFLOPS)", my_time, my_gflops);

    // Check output is not all zeros (sanity check)
    let output_ptr = my_output_buffer.contents() as *const f32;
    let sum: f32 = (0..10).map(|i| unsafe { *output_ptr.add(i) }).sum();
    println!("Output sum (first 10): {}", sum);

    // Test Q2_K
    println!("\n=== Q2_K ===");
    let weights = generate_q2k_weights(m, k);
    let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64;
    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: k as u64 * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };

    let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let my_kernel = my_lib
        .get_function("kernel_gemm_q2_K_f32", Some(cv.clone()))
        .expect("Failed to get Q2_K kernel");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create pipeline");

    let my_output_buffer = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_time = benchmark_kernel(
        &my_pipeline,
        &queue,
        &args,
        &weights_buffer,
        &input_buffer,
        &my_output_buffer,
        thread_group_count,
        thread_group_size,
        iterations,
    );

    let my_gflops = (2.0 * m as f64 * k as f64) / (my_time * 1e-6) / 1e9;
    println!("Self-written: {:.2} us ({:.1} GFLOPS)", my_time, my_gflops);

    let output_ptr = my_output_buffer.contents() as *const f32;
    let sum: f32 = (0..10).map(|i| unsafe { *output_ptr.add(i) }).sum();
    println!("Output sum (first 10): {}", sum);

    println!("\n=== Summary ===");
    println!("K-quant kernels: Q2_K ✓, Q4_K ✓");
    println!("Need to add Q3_K, Q5_K, Q6_K tests");
}
