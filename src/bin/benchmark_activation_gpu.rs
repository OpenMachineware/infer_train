// Benchmark GPU activation functions
use metal::{Device, CompileOptions, MTLSize};

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Compile shader
    let source = include_str!("../../shaders/activation.metal");
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &compile_options).expect("Compile failed");

    let silu_kernel = library.get_function("kernel_silu_f32", None).unwrap();
    let gelu_kernel = library.get_function("kernel_gelu_f32", None).unwrap();
    let quick_kernel = library.get_function("kernel_gelu_quick_f32", None).unwrap();

    let silu_4_kernel = library.get_function("kernel_silu_f32_4", None).unwrap();
    let gelu_4_kernel = library.get_function("kernel_gelu_f32_4", None).unwrap();
    let quick_4_kernel = library.get_function("kernel_gelu_quick_f32_4", None).unwrap();

    let silu_pipeline = device.new_compute_pipeline_state_with_function(&silu_kernel).unwrap();
    let gelu_pipeline = device.new_compute_pipeline_state_with_function(&gelu_kernel).unwrap();
    let quick_pipeline = device.new_compute_pipeline_state_with_function(&quick_kernel).unwrap();

    let silu_4_pipeline = device.new_compute_pipeline_state_with_function(&silu_4_kernel).unwrap();
    let gelu_4_pipeline = device.new_compute_pipeline_state_with_function(&gelu_4_kernel).unwrap();
    let quick_4_pipeline = device.new_compute_pipeline_state_with_function(&quick_4_kernel).unwrap();

    let sizes = [1024, 4096, 16384, 65536, 262144, 1048576];

    println!("=== GPU Activation Benchmark ===\n");
    println!("Size      | SiLU (GB/s) | GELU (GB/s) | Quick (GB/s) | SiLU-4 | GELU-4 | Quick-4");
    println!("----------+-------------+-------------+--------------+--------+--------+--------");

    for &size in &sizes {
        let input = vec![1.0f32; size];
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device.new_buffer(
            (size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        run_kernel(&queue, &silu_pipeline, &input_buffer, &output_buffer, size, 10);

        // Benchmark scalar kernels
        let silu_time = run_kernel(&queue, &silu_pipeline, &input_buffer, &output_buffer, size, 100);
        let gelu_time = run_kernel(&queue, &gelu_pipeline, &input_buffer, &output_buffer, size, 100);
        let quick_time = run_kernel(&queue, &quick_pipeline, &input_buffer, &output_buffer, size, 100);

        // Benchmark vec4 kernels
        let silu_4_time = run_kernel(&queue, &silu_4_pipeline, &input_buffer, &output_buffer, size/4, 100);
        let gelu_4_time = run_kernel(&queue, &gelu_4_pipeline, &input_buffer, &output_buffer, size/4, 100);
        let quick_4_time = run_kernel(&queue, &quick_4_pipeline, &input_buffer, &output_buffer, size/4, 100);

        let bytes = (size * 4 * 2) as f64; // read + write

        println!(
            "{:8} | {:8.2}    | {:8.2}    | {:8.2}     | {:6.2} | {:6.2} | {:6.2}",
            size,
            bytes / silu_time / 1e9,
            bytes / gelu_time / 1e9,
            bytes / quick_time / 1e9,
            bytes / silu_4_time / 1e9,
            bytes / gelu_4_time / 1e9,
            bytes / quick_4_time / 1e9,
        );
    }
}

fn run_kernel(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineState,
    input: &metal::BufferRef,
    output: &metal::BufferRef,
    thread_count: usize,
    iterations: usize,
) -> f64 {
    use std::time::Instant;

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);

    let threadgroup_size = MTLSize { width: 256, height: 1, depth: 1 };
    let grid_size = MTLSize { width: thread_count as u64, height: 1, depth: 1 };
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    start.elapsed().as_secs_f64() / iterations as f64
}
