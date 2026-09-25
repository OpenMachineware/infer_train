// Benchmark GPU execution time only (matching llama.cpp methodology)
// Run: cargo run --release --bin benchmark_gpu_only

use infer_train::quant::types::BlockQ4K;
use metal::{Device, MTLSize, ComputePipelineState, Buffer, CommandQueue};
use half::f16;
use std::time::Instant;

fn main() {
    println!("=== GPU Execution Time Benchmark (matching llama.cpp) ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let m = 4096;
    let k = 4096;
    let n = 128;

    println!("M={}, K={}, N={}", m, k, n);
    println!("Operations: {:.2} GFLOPS\n", (2.0 * m as f64 * k as f64 * n as f64) / 1e9);

    // Create weights
    let weights: Vec<BlockQ4K> = (0..m * k / 256).map(|i| {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for j in 0..12 { scales[j] = (40 + j as u8 % 8) as u8; }
        for j in 0..128 { qs[j] = ((i * 17 + j * 3) % 256) as u8; }
        BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        }
    }).collect();

    // Create input (pre-convert to FP16)
    let input_f16: Vec<u16> = (0..k * n)
        .map(|i| half::f16::from_f32((i % 100) as f32 * 0.01 - 0.5).to_bits())
        .collect();

    // Pre-allocate buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        (weights.len() * std::mem::size_of::<BlockQ4K>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const std::ffi::c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (m * n * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Compile template kernel from llama.cpp
    let shader_source = include_str!("../../shaders/mul_mm_standalone.metal");
    let compile_options = metal::CompileOptions::new();
    let library = device.new_library_with_source(shader_source, &compile_options)
        .expect("Failed to compile shader");
    let kernel = library.get_function("kernel_mul_mm_q4_K_f32", None)
        .expect("Failed to get kernel");
    let template_pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    // Args struct (matching llama.cpp's ggml_metal_kargs_mul_mm)
    #[repr(C)]
    struct GemmArgs {
        ne00: i32,   // K
        ne02: i32,   // batch dim
        nb01: u64,   // row stride for A
        nb02: u64,   // batch stride
        nb03: u64,   // batch3 stride
        ne12: i32,   // batch dim for B
        nb10: u64,   // element stride for B
        nb11: u64,   // row stride for B
        nb12: u64,   // batch stride
        nb13: u64,   // batch3 stride
        ne0: i32,    // M (output rows)
        ne1: i32,    // N (output cols)
        r2: i16,
        r3: i16,
    }

    let nb = k / 256;
    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: (nb * std::mem::size_of::<BlockQ4K>()) as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    let args_buffer = device.new_buffer_with_data(
        &args as *const GemmArgs as *const std::ffi::c_void,
        std::mem::size_of::<GemmArgs>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    run_gemm(&queue, &template_pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer);
    run_gemm(&queue, &template_pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer);

    // Benchmark template kernel
    let n_iter = 10;
    let start = Instant::now();
    for _ in 0..n_iter {
        run_gemm(&queue, &template_pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer);
    }
    let template_time = start.elapsed().as_micros() as f64 / n_iter as f64;

    // Calculate GFLOPS
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let template_gflops = flops / (template_time * 1e-6) / 1e9;

    println!("┌────────────────────┬────────────┬──────────┐");
    println!("│ Kernel             │ Time (µs)  │ GFLOPS   │");
    println!("├────────────────────┼────────────┼──────────┤");
    println!("│ Template (llama.cpp) │ {:10.2} │ {:8.2} │", template_time, template_gflops);
    println!("└────────────────────┴────────────┴──────────┘");
    println!();

    // Compare with llama.cpp reference
    println!("llama.cpp Q4_K: ~3656 GFLOPS (M1 Max, N=128 batch)");
    println!("Our template:   {:.0} GFLOPS ({:.1}% of llama.cpp)", template_gflops, template_gflops / 3656.0 * 100.0);
}

fn run_gemm(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args: &Buffer,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
) {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args), 0);
    encoder.set_buffer(1, Some(weights), 0);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_threadgroup_memory_length(0, 8192);

    // Grid: x=4 (N/32 columns), y=64 (M/64 rows)
    // Kernel uses: r0 = tgpig.y * 64, r1 = tgpig.x * 32
    let grid = MTLSize::new(4, 64, 1);
    let threadgroup = MTLSize::new(32, 4, 1);

    encoder.dispatch_thread_groups(grid, threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
}
