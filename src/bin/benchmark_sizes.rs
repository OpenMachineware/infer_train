// Benchmark different matrix sizes for template GEMM
// Run: cargo run --release --bin benchmark_sizes

use metal::{Device, MTLSize, CompileOptions, FunctionDescriptor, FunctionConstantValues, MTLDataType};

fn main() {
    println!("=== Template GEMM Size Variations ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // Compile shader
    let shader_source = include_str!("../../shaders/mul_mm_standalone.metal");
    let compile_options = CompileOptions::new();
    let library = device.new_library_with_source(shader_source, &compile_options)
        .expect("Failed to compile shader");

    // Test scenarios: (M, N, K) - typical for LLM inference
    let scenarios = [
        ("Decode N=1", 4096, 1, 4096),
        ("Decode N=4", 4096, 4, 4096),
        ("Decode N=8", 4096, 8, 4096),
        ("Small batch", 4096, 16, 4096),
        ("Medium batch", 4096, 32, 4096),
        ("Large batch", 4096, 64, 4096),
        ("XL batch", 4096, 128, 4096),
        ("Small M/K", 1024, 16, 1024),
        ("Large M/K", 8192, 16, 8192),
    ];

    // Block sizes for each format
    let formats = [
        ("Q2_K", "kernel_mul_mm_q2_K_f32", 144),
        ("Q3_K", "kernel_mul_mm_q3_K_f32", 128),
        ("Q4_K", "kernel_mul_mm_q4_K_f32", 144),
        ("Q5_K", "kernel_mul_mm_q5_K_f32", 176),
        ("Q6_K", "kernel_mul_mm_q6_K_f32", 210),
    ];

    println!("┌──────────────┬───────┬──────┬──────┬──────────┬──────────┐");
    println!("│ Scenario     │   M   │   N  │   K  │ GFLOPS   │ vs llama │");
    println!("├──────────────┼───────┼──────┼──────┼──────────┼──────────┤");

    for (format_name, kernel_name, block_size) in formats.iter() {
        println!("│ {}                                                                │", format_name);

        for (scenario_name, m, n, k) in scenarios.iter() {
            let gflops = benchmark_gemm(&device, &queue, &library, *m, *n, *k, *block_size, kernel_name);

            // Estimate llama.cpp baseline
            let llama_baseline = estimate_llama_baseline(*m, *n, *k);
            let ratio = gflops / llama_baseline * 100.0;

            println!("│ {:12} │ {:5} │ {:4} │ {:4} │ {:8.2} │ {:7.1}% │",
                scenario_name, m, n, k, gflops, ratio);
        }
        println!("├──────────────┼───────┼──────┼──────┼──────────┼──────────┤");
    }
    println!("└──────────────┴───────┴──────┴──────┴──────────┴──────────┘");
}

fn estimate_llama_baseline(m: usize, n: usize, k: usize) -> f64 {
    // Based on measurements: llama.cpp achieves ~2800 GFLOPS for large batches
    let efficiency = if n <= 8 {
        0.35 + 0.05 * n as f64 // N=1: 40%, N=8: 75%
    } else if n <= 32 {
        0.75 + 0.005 * (n - 8) as f64 // N=32: ~85%
    } else {
        0.85 + 0.0007 * (n - 32) as f64 // N=128: ~92%
    };

    2800.0 * efficiency
}

fn benchmark_gemm(
    device: &Device,
    queue: &metal::CommandQueue,
    library: &metal::Library,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
    kernel_name: &str,
) -> f64 {
    // Create function constants
    let constant_values = FunctionConstantValues::new();
    let bc_out = m % 64 != 0 || n % 32 != 0;
    let false_val: bool = false;
    let bc_out_val: bool = bc_out;
    let one_val: i16 = 1;

    unsafe {
        constant_values.set_constant_value_at_index(
            &false_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 0);
        constant_values.set_constant_value_at_index(
            &bc_out_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 1);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 2);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 3);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 4);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 5);
    }

    let descriptor = FunctionDescriptor::new();
    descriptor.set_name(kernel_name);
    descriptor.set_constant_values(&constant_values);

    let function = match library.new_function_with_descriptor(&descriptor) {
        Ok(f) => f,
        Err(_) => return 0.0,
    };

    let pipeline = device.new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline");

    // Allocate buffers
    let num_blocks_k = (k + 255) / 256;
    let weights_size = m * num_blocks_k * block_size;
    let input_size = n * k * 2; // FP16
    let output_size = m * n * 4; // FP32

    let weights_buffer = device.new_buffer(weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let input_buffer = device.new_buffer(input_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

    // Args
    #[repr(C)]
    struct GemmArgs {
        ne00: i32, ne02: i32, nb01: u64, nb02: u64, nb03: u64,
        ne12: i32, nb10: u64, nb11: u64, nb12: u64, nb13: u64,
        ne0: i32, ne1: i32, r2: i16, r3: i16,
    }

    let args = GemmArgs {
        ne00: k as i32, ne02: 1, nb01: (num_blocks_k * block_size) as u64, nb02: 0, nb03: 0,
        ne12: 1, nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
        ne0: m as i32, ne1: n as i32, r2: 1, r3: 1,
    };
    let args_buffer = device.new_buffer_with_data(
        &args as *const GemmArgs as *const std::ffi::c_void,
        std::mem::size_of::<GemmArgs>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Grid: x = N/32 (columns), y = M/64 (rows)
    let grid_x = (n + 31) / 32;
    let grid_y = (m + 63) / 64;
    let grid_size = MTLSize::new(grid_x as u64, grid_y as u64, 1);
    let threadgroup_size = MTLSize::new(32, 4, 1);

    // Warmup
    for _ in 0..10 {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&args_buffer), 0);
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_threadgroup_memory_length(4096 + 2048, 0);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Benchmark
    let iterations = 20;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&args_buffer), 0);
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_threadgroup_memory_length(4096 + 2048, 0);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * n as f64 * k as f64 * iterations as f64;
    flops / elapsed / 1e9
}
