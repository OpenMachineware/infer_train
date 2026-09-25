// Benchmark optimized kernel with hardcoded constants
// Run: cargo run --release --bin benchmark_optimized

use infer_train::quant::types::BlockQ4K;
use metal::{Device, MTLSize, CompileOptions, FunctionDescriptor, FunctionConstantValues, MTLDataType};
use half::f16;

fn main() {
    println!("=== Optimized GEMM with Hardcoded Constants ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let m = 4096;
    let k = 4096;
    let n = 128;

    println!("M={}, K={}, N={}\n", m, k, n);

    // Create weights
    let weights: Vec<BlockQ4K> = (0..m * k / 256).map(|i| {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for j in 0..12 { scales[j] = ((i * 17 + j + 13) % 256) as u8; }
        for j in 0..128 { qs[j] = ((i * 17 + j + 13) % 256) as u8; }
        BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        }
    }).collect();

    // Create input
    let input_f16: Vec<u16> = (0..k * n)
        .map(|i| half::f16::from_f32((i % 10) as f32 * 0.1).to_bits())
        .collect();

    // Create buffers
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

    // Compile both kernels
    let opt_shader = include_str!("../../shaders/mul_mm_optimized.metal");
    let orig_shader = include_str!("../../shaders/mul_mm_standalone.metal");
    let compile_options = CompileOptions::new();

    let opt_library = device.new_library_with_source(opt_shader, &compile_options)
        .expect("Failed to compile optimized shader");
    let orig_library = device.new_library_with_source(orig_shader, &compile_options)
        .expect("Failed to compile original shader");

    let opt_kernel = opt_library.get_function("kernel_mul_mm_q4_K_opt", None)
        .expect("Failed to get optimized kernel");
    
    // For original kernel with function constants, we need to specialize it
    let constant_values = FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
    
    unsafe {
        constant_values.set_constant_value_at_index(
            &false_val as *const _ as *const std::ffi::c_void,
            MTLDataType::Bool,
            0
        );
        constant_values.set_constant_value_at_index(
            &false_val as *const _ as *const std::ffi::c_void,
            MTLDataType::Bool,
            1
        );
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void,
            MTLDataType::Short,
            2
        );
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void,
            MTLDataType::Short,
            3
        );
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void,
            MTLDataType::Short,
            4
        );
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void,
            MTLDataType::Short,
            5
        );
    }
    
    let descriptor = FunctionDescriptor::new();
    descriptor.set_name("kernel_mul_mm_q4_K_f32");
    descriptor.set_constant_values(&constant_values);
    
    let orig_kernel = orig_library.new_function_with_descriptor(&descriptor)
        .expect("Failed to create specialized original kernel");

    let opt_pipeline = device.new_compute_pipeline_state_with_function(&opt_kernel)
        .expect("Failed to create optimized pipeline");
    let orig_pipeline = device.new_compute_pipeline_state_with_function(&orig_kernel)
        .expect("Failed to create original pipeline");

    println!("Optimized pipeline max threads: {}", opt_pipeline.max_total_threads_per_threadgroup());
    println!("Original pipeline max threads: {}\n", orig_pipeline.max_total_threads_per_threadgroup());

    // Args
    #[repr(C)]
    struct GemmArgs {
        ne00: i32, ne02: i32, nb01: u64, nb02: u64, nb03: u64,
        ne12: i32, nb10: u64, nb11: u64, nb12: u64, nb13: u64,
        ne0: i32, ne1: i32, r2: i16, r3: i16,
    }

    let nb = k / 256;
    let args = GemmArgs {
        ne00: k as i32, ne02: 1,
        nb01: (nb * std::mem::size_of::<BlockQ4K>()) as u64,
        nb02: 0, nb03: 0,
        ne12: 1, nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
        ne0: m as i32, ne1: n as i32, r2: 1, r3: 1,
    };

    let args_buffer = device.new_buffer_with_data(
        &args as *const GemmArgs as *const std::ffi::c_void,
        std::mem::size_of::<GemmArgs>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    for _ in 0..2 {
        run_gemm(&queue, &opt_pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer);
        run_gemm(&queue, &orig_pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer);
    }

    // Benchmark
    let n_iter = 10;

    let opt_time = benchmark(&queue, &opt_pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer, n_iter);
    let orig_time = benchmark(&queue, &orig_pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer, n_iter);

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let opt_gflops = flops / (opt_time * 1e-6) / 1e9;
    let orig_gflops = flops / (orig_time * 1e-6) / 1e9;

    println!("┌──────────────────────────┬────────────┬──────────┬───────────┐");
    println!("│ Kernel                   │ Time (µs)  │ GFLOPS   │ vs llama  │");
    println!("├──────────────────────────┼────────────┼──────────┼───────────┤");
    println!("│ Optimized (hardcoded)    │ {:10.2} │ {:8.2} │ {:8.1}% │", opt_time, opt_gflops, opt_gflops / 3552.0 * 100.0);
    println!("│ Original (FC)            │ {:10.2} │ {:8.2} │ {:8.1}% │", orig_time, orig_gflops, orig_gflops / 3552.0 * 100.0);
    println!("└──────────────────────────┴────────────┴──────────┴───────────┘");
    println!();
    println!("llama.cpp Q4_K: ~3552 GFLOPS");

    // Verify output
    let output = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, m * n)
    };
    let non_zero = output.iter().filter(|&&x| x != 0.0).count();
    println!("\nOutput coverage: {} / {} ({:.1}%)", non_zero, m * n, 100.0 * non_zero as f64 / (m * n) as f64);
}

fn benchmark(
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    args: &metal::Buffer,
    weights: &metal::Buffer,
    input: &metal::Buffer,
    output: &metal::Buffer,
    n_iter: usize,
) -> f64 {
    let start = std::time::Instant::now();
    for _ in 0..n_iter {
        run_gemm(queue, pipeline, args, weights, input, output);
    }
    start.elapsed().as_micros() as f64 / n_iter as f64
}

fn run_gemm(
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    args: &metal::Buffer,
    weights: &metal::Buffer,
    input: &metal::Buffer,
    output: &metal::Buffer,
) {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args), 0);
    encoder.set_buffer(1, Some(weights), 0);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_threadgroup_memory_length(0, 8192);

    let grid = MTLSize::new(4, 64, 1);
    let threadgroup = MTLSize::new(32, 4, 1);

    encoder.dispatch_thread_groups(grid, threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
}