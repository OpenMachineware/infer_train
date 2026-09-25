// Benchmark our GPU activation vs llama.cpp's GPU activation
// We create standalone versions of llama.cpp kernels to avoid header dependency issues
use metal::{Device, CompileOptions, MTLSize};
use std::time::Instant;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Compile our activation library
    let our_source = include_str!("../../shaders/activation.metal");
    let our_library = device.new_library_with_source(our_source, &compile_options).expect("Our compile failed");

    // Create llama.cpp style standalone kernels (copied from llama.cpp to avoid dependency issues)
    let llama_source = r#"
#include <metal_stdlib>
using namespace metal;

constant float GELU_COEF_A     = 0.044715f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float GELU_QUICK_COEF = -1.702f;

// llama.cpp kernel_unary_f32_f32 for SILU (cnt=false path, non-contiguous)
kernel void kernel_llama_silu_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) {
        return;
    }
    const float x = src0[i0];
    dst[i0] = x / (1.0f + exp(-x));
}

// llama.cpp kernel_unary_f32_f32 for GELU
kernel void kernel_llama_gelu_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) {
        return;
    }
    const float x = src0[i0];
    dst[i0] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// llama.cpp kernel_unary_f32_f32 for GELU_QUICK
kernel void kernel_llama_gelu_quick_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) {
        return;
    }
    const float x = src0[i0];
    dst[i0] = x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));
}
"#;

    let llama_library = device.new_library_with_source(llama_source, &compile_options).expect("llama.cpp compile failed");

    // Create our pipelines
    let our_silu = create_pipeline(&device, &our_library, "kernel_silu_f32");
    let our_gelu = create_pipeline(&device, &our_library, "kernel_gelu_f32");
    let our_quick = create_pipeline(&device, &our_library, "kernel_gelu_quick_f32");

    // Create llama.cpp pipelines
    let llama_silu = create_pipeline(&device, &llama_library, "kernel_llama_silu_f32");
    let llama_gelu = create_pipeline(&device, &llama_library, "kernel_llama_gelu_f32");
    let llama_quick = create_pipeline(&device, &llama_library, "kernel_llama_gelu_quick_f32");

    let sizes = [1024, 4096, 16384, 65536, 262144, 1048576];

    println!("=== GPU Activation: Our vs llama.cpp ===\n");
    println!("Size      | Our SiLU us | llama SiLU | Ratio | Our GELU | llama GELU | Ratio | Our Quick | llama Quick | Ratio");
    println!("----------+-------------+------------+-------+----------+------------+-------+-----------+-------------+-------");

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
        run_our_kernel(&queue, &our_silu, &input_buffer, &output_buffer, size, 5);
        run_llama_kernel(&queue, &llama_silu, &input_buffer, &output_buffer, size, 5);

        // Benchmark - our kernels
        let our_silu_time = run_our_kernel(&queue, &our_silu, &input_buffer, &output_buffer, size, 100);
        let our_gelu_time = run_our_kernel(&queue, &our_gelu, &input_buffer, &output_buffer, size, 100);
        let our_quick_time = run_our_kernel(&queue, &our_quick, &input_buffer, &output_buffer, size, 100);

        // Benchmark - llama.cpp kernels
        let llama_silu_time = run_llama_kernel(&queue, &llama_silu, &input_buffer, &output_buffer, size, 100);
        let llama_gelu_time = run_llama_kernel(&queue, &llama_gelu, &input_buffer, &output_buffer, size, 100);
        let llama_quick_time = run_llama_kernel(&queue, &llama_quick, &input_buffer, &output_buffer, size, 100);

        let silu_ratio = llama_silu_time / our_silu_time * 100.0;
        let gelu_ratio = llama_gelu_time / our_gelu_time * 100.0;
        let quick_ratio = llama_quick_time / our_quick_time * 100.0;

        println!(
            "{:8} | {:8.1}   | {:8.1}   | {:4.0}% | {:8.1} | {:8.1}   | {:4.0}% | {:8.1}  | {:8.1}    | {:4.0}%",
            size,
            our_silu_time * 1e6,
            llama_silu_time * 1e6,
            silu_ratio,
            our_gelu_time * 1e6,
            llama_gelu_time * 1e6,
            gelu_ratio,
            our_quick_time * 1e6,
            llama_quick_time * 1e6,
            quick_ratio,
        );
    }
}

fn create_pipeline(device: &Device, library: &metal::LibraryRef, name: &str) -> metal::ComputePipelineState {
    let kernel = library.get_function(name, None).expect(&format!("Failed to get {}", name));
    device.new_compute_pipeline_state_with_function(&kernel).expect(&format!("Failed to create pipeline for {}", name))
}

fn run_our_kernel(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineState,
    input: &metal::BufferRef,
    output: &metal::BufferRef,
    n: usize,
    iterations: usize,
) -> f64 {
    let threadgroup_size = MTLSize { width: 256, height: 1, depth: 1 };
    let grid_size = MTLSize { width: n as u64, height: 1, depth: 1 };

    // Warmup run
    {
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

fn run_llama_kernel(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineState,
    input: &metal::BufferRef,
    output: &metal::BufferRef,
    n: usize,
    iterations: usize,
) -> f64 {
    let ne00 = n as i32;
    let threads_per_group = 256;
    let threadgroup_size = MTLSize { width: threads_per_group as u64, height: 1, depth: 1 };
    let num_groups = (n + threads_per_group - 1) / threads_per_group;
    let grid_size = MTLSize { width: num_groups as u64, height: 1, depth: 1 };

    // Warmup run
    {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    start.elapsed().as_secs_f64() / iterations as f64
}
