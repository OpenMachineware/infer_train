// Benchmark all K-quant MV kernels vs llama.cpp
// Run: cargo run --release --bin benchmark_mv_all_formats

use metal::{Device, MTLSize, CompileOptions};

fn main() {
    println!("=== K-Quant MV Performance (All Formats) ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let formats = [
        ("Q2_K", 144, "kernel_mul_mv_q2_K_f32", "shaders/mv_q2_k.metal"),
        ("Q3_K", 128, "kernel_mul_mv_q3_K_f32", "shaders/mv_q3_k.metal"),
        ("Q4_K", 144, "kernel_mul_mv_q4_K_f32", "shaders/mv_q4_k.metal"),
        ("Q5_K", 176, "kernel_mul_mv_q5_K_f32", "shaders/mv_q5_k.metal"),
        ("Q6_K", 210, "kernel_mul_mv_q6_K_f32", "shaders/mv_q6_k.metal"),
    ];

    // Test sizes: small, medium, large
    let sizes = [
        ("Small", 1024, 1024),
        ("Medium", 2048, 2048),
        ("Large", 4096, 4096),
    ];

    println!("┌─────────┬────────┬──────────┬──────────┬──────────┐");
    println!("│ Format  │ Size   │ Our      │ llama.cpp│ Ratio    │");
    println!("├─────────┼────────┼──────────┼──────────┼──────────┤");

    // llama.cpp baselines (from test-kquant-metal, M=K=4096)
    let llama_baselines = [
        ("Q2_K", 93.0),
        ("Q3_K", 54.0),
        ("Q4_K", 65.0),
        ("Q5_K", 75.0),
        ("Q6_K", 73.0),
    ];

    for (format_name, block_size, kernel_name, shader_path) in formats.iter() {
        // Load and compile shader
        let shader_source = std::fs::read_to_string(shader_path)
            .expect(&format!("Failed to read {}", shader_path));
        let compile_options = CompileOptions::new();
        compile_options.set_fast_math_enabled(true);
        let library = device.new_library_with_source(&shader_source, &compile_options)
            .expect("Failed to compile shader");
        let function = library.get_function(kernel_name, None).expect("Failed to get function");
        let pipeline = device.new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create pipeline");

        // Get llama.cpp baseline
        let llama_baseline = llama_baselines.iter()
            .find(|(name, _)| name == format_name)
            .map(|(_, gflops)| *gflops)
            .unwrap_or(70.0);

        for (size_name, m, k) in sizes.iter() {
            let gflops = benchmark_mv_format(&device, &queue, &pipeline, *m, *k, *block_size);
            let ratio = gflops / llama_baseline * 100.0;

            println!("│ {:7} │ {:6} │ {:8.2} │ {:8.1} │ {:7.1}% │",
                format_name, size_name, gflops, llama_baseline, ratio);
        }
    }

    println!("└─────────┴────────┴──────────┴──────────┴──────────┘");
}

fn benchmark_mv_format(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    m: usize,
    k: usize,
    block_size: usize,
) -> f64 {
    let num_blocks_k = (k + 255) / 256;
    let weights_size = m * num_blocks_k * block_size;
    let input_size = k * 4;
    let output_size = m * 4;

    let weights_buffer = device.new_buffer(weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let input_buffer = device.new_buffer(input_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct MvArgs {
        ne00: u32,
        ne01: u32,
        nb01: u64,
    }

    let args = MvArgs {
        ne00: k as u32,
        ne01: m as u32,
        nb01: (num_blocks_k * block_size) as u64,
    };
    let args_buffer = device.new_buffer_with_data(
        &args as *const MvArgs as *const std::ffi::c_void,
        std::mem::size_of::<MvArgs>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // NSG and NR0 depend on format - use common defaults
    let nsg = 2;
    let nr0 = 2;
    let rows_per_tg = nr0 * nsg;
    let grid_x = (m + rows_per_tg - 1) / rows_per_tg;
    let grid_size = MTLSize::new(grid_x as u64, 1, 1);
    let threadgroup_size = MTLSize::new(32, nsg as u64, 1);

    // Warmup
    for _ in 0..20 {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&args_buffer), 0);
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
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&args_buffer), 0);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * k as f64 * iterations as f64;
    flops / elapsed / 1e9
}
