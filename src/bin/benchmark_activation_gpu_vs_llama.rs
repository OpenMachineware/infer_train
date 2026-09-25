// Benchmark our GPU activation vs llama.cpp's GPU activation
// Conclusion: Both implementations achieve parity because kernels are identical
// - Same formulas: SiLU = x / (1 + exp(-x)), GELU = 0.5 * x * (1 + tanh(...))
// - Same Metal built-in functions: exp(), precise::tanh()
// - Memory-bound element-wise operations
// - No optimization opportunity beyond Metal compiler's capabilities
use metal::{Device, CompileOptions, MTLSize};
use std::time::Instant;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Our activation kernels (from shaders/activation.metal)
    // Uses dispatch_threads with thread_position_in_grid - simpler and more efficient
    let our_source = include_str!("../../shaders/activation.metal");
    let our_library = device.new_library_with_source(our_source, &compile_options).expect("Our library compile failed");

    // llama.cpp style kernels - TWO modes: cnt and non-cnt
    let llama_source = r#"
#include <metal_stdlib>
using namespace metal;

constant float GELU_COEF_A     = 0.044715f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float GELU_QUICK_COEF = -1.702f;

// llama.cpp CONTIGUOUS mode kernel (cnt=true)
// Dispatch: n threadgroups, 1 thread per threadgroup
// Used for n < 32768
kernel void kernel_llama_cnt_silu_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x;  // Direct threadgroup index
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = x / (1.0f + exp(-x));
}

kernel void kernel_llama_cnt_gelu_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

kernel void kernel_llama_cnt_gelu_quick_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));
}

// llama.cpp NON-CONTIGUOUS mode kernel (cnt=false)
// Dispatch: (n+255)/256 threadgroups, 256 threads per threadgroup
// Used for n >= 32768
// This is the ACTUAL llama.cpp kernel for large sizes
kernel void kernel_llama_silu_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    // llama.cpp uses: i0 = k0*ntg.x + tpitg.x where k0 = tgpig.x/ne01 (for 1D, ne01=1)
    // Simplified for 1D contiguous: i0 = tgpig.x * ntg.x + tpitg.x
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = x / (1.0f + exp(-x));
}

kernel void kernel_llama_gelu_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

kernel void kernel_llama_gelu_quick_f32(
    constant int & ne00,
    device const float * src0,
    device float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));
}

// llama.cpp float4 kernels (c4=true) for better throughput
kernel void kernel_llama_silu_f32_4(
    constant int & ne00,
    device const float4 * src0,
    device float4 * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float4 x = src0[i0];
    dst[i0] = x / (1.0f + exp(-x));
}

kernel void kernel_llama_gelu_f32_4(
    constant int & ne00,
    device const float4 * src0,
    device float4 * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float4 x = src0[i0];
    dst[i0] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

kernel void kernel_llama_gelu_quick_f32_4(
    constant int & ne00,
    device const float4 * src0,
    device float4 * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3 ntg[[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float4 x = src0[i0];
    dst[i0] = x / (1.0f + exp(GELU_QUICK_COEF * x));
}
"#;

    let llama_library = device.new_library_with_source(llama_source, &compile_options).expect("llama.cpp library compile failed");

    // Create pipelines
    // Our kernels: hybrid dispatch (dispatch_threads for small, dispatch_thread_groups with float4 for large)
    let our_silu = create_pipeline(&device, &our_library, "kernel_silu_f32");
    let our_silu_tg = create_pipeline(&device, &our_library, "kernel_silu_f32_tg");
    let our_silu_4_tg = create_pipeline(&device, &our_library, "kernel_silu_f32_4_tg");
    let our_gelu = create_pipeline(&device, &our_library, "kernel_gelu_f32");
    let our_gelu_tg = create_pipeline(&device, &our_library, "kernel_gelu_f32_tg");
    let our_gelu_4_tg = create_pipeline(&device, &our_library, "kernel_gelu_f32_4_tg");
    let our_quick = create_pipeline(&device, &our_library, "kernel_gelu_quick_f32");
    let our_quick_tg = create_pipeline(&device, &our_library, "kernel_gelu_quick_f32_tg");
    let our_quick_4_tg = create_pipeline(&device, &our_library, "kernel_gelu_quick_f32_4_tg");

    // llama.cpp kernels: cnt mode for small sizes, non-cnt for large sizes
    let llama_cnt_silu = create_pipeline(&device, &llama_library, "kernel_llama_cnt_silu_f32");
    let llama_cnt_gelu = create_pipeline(&device, &llama_library, "kernel_llama_cnt_gelu_f32");
    let llama_cnt_quick = create_pipeline(&device, &llama_library, "kernel_llama_cnt_gelu_quick_f32");
    // llama.cpp non-cnt kernels (for n >= 32768)
    let llama_silu = create_pipeline(&device, &llama_library, "kernel_llama_silu_f32");
    let llama_gelu = create_pipeline(&device, &llama_library, "kernel_llama_gelu_f32");
    let llama_quick = create_pipeline(&device, &llama_library, "kernel_llama_gelu_quick_f32");
    // llama.cpp float4 kernels (for n >= 32768 and n % 4 == 0)
    let llama_silu_4 = create_pipeline(&device, &llama_library, "kernel_llama_silu_f32_4");
    let llama_gelu_4 = create_pipeline(&device, &llama_library, "kernel_llama_gelu_f32_4");
    let llama_quick_4 = create_pipeline(&device, &llama_library, "kernel_llama_gelu_quick_f32_4");

    println!("Warming up GPU...");
    warmup_gpu(&queue, &[
        &our_silu, &our_silu_tg, &our_silu_4_tg,
        &our_gelu, &our_gelu_tg, &our_gelu_4_tg,
        &our_quick, &our_quick_tg, &our_quick_4_tg,
        &llama_cnt_silu, &llama_cnt_gelu, &llama_cnt_quick,
        &llama_silu, &llama_gelu, &llama_quick,
        &llama_silu_4, &llama_gelu_4, &llama_quick_4,
    ]);

    let sizes = [1024, 4096, 16384, 65536, 262144, 1048576];
    let runs = 10;
    let dispatches_per_run = 100;
    const DISPATCH_THRESHOLD: usize = 32768;

    println!("\n=== GPU Activation: Our (hybrid + float4) vs llama.cpp ===");
    println!("Our: dispatch_threads if n < 32768, else dispatch_thread_groups with float4");
    println!("llama.cpp: 1 thread per group if n < 32768, else 256 threads per group\n");
    println!("Size      | Our SiLU   | llama SiLU | Ratio | Our GELU   | llama GELU | Ratio | Our Quick  | llama Quick | Ratio");
    println!("----------+------------+------------+-------+------------+------------+-------+------------+-------------+-------");

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

        // Select our pipeline based on size
        let (our_silu_pipe, our_gelu_pipe, our_quick_pipe, use_float4) = if size < DISPATCH_THRESHOLD {
            (&our_silu, &our_gelu, &our_quick, false)
        } else if size % 4 == 0 {
            (&our_silu_4_tg, &our_gelu_4_tg, &our_quick_4_tg, true)
        } else {
            (&our_silu_tg, &our_gelu_tg, &our_quick_tg, false)
        };

        // Select llama.cpp pipeline based on size
        // - n < 32768: cnt mode (1 thread per group)
        // - n >= 32768: non-cnt mode (256 threads per group, use float4 if possible)
        let (llama_silu_pipe, llama_gelu_pipe, llama_quick_pipe, llama_use_float4) = if size < DISPATCH_THRESHOLD {
            (&llama_cnt_silu, &llama_cnt_gelu, &llama_cnt_quick, false)
        } else if size % 4 == 0 {
            (&llama_silu_4, &llama_gelu_4, &llama_quick_4, true)
        } else {
            (&llama_silu, &llama_gelu, &llama_quick, false)
        };

        let our_silu_time = run_benchmark(&queue, our_silu_pipe, llama_silu_pipe, &input_buffer, &output_buffer, size, dispatches_per_run, runs, true, use_float4);
        let llama_silu_time = run_benchmark(&queue, our_silu_pipe, llama_silu_pipe, &input_buffer, &output_buffer, size, dispatches_per_run, runs, false, llama_use_float4);

        let our_gelu_time = run_benchmark(&queue, our_gelu_pipe, llama_gelu_pipe, &input_buffer, &output_buffer, size, dispatches_per_run, runs, true, use_float4);
        let llama_gelu_time = run_benchmark(&queue, our_gelu_pipe, llama_gelu_pipe, &input_buffer, &output_buffer, size, dispatches_per_run, runs, false, llama_use_float4);

        let our_quick_time = run_benchmark(&queue, our_quick_pipe, llama_quick_pipe, &input_buffer, &output_buffer, size, dispatches_per_run, runs, true, use_float4);
        let llama_quick_time = run_benchmark(&queue, our_quick_pipe, llama_quick_pipe, &input_buffer, &output_buffer, size, dispatches_per_run, runs, false, llama_use_float4);

        let silu_ratio = llama_silu_time / our_silu_time * 100.0;
        let gelu_ratio = llama_gelu_time / our_gelu_time * 100.0;
        let quick_ratio = llama_quick_time / our_quick_time * 100.0;

        println!(
            "{:8} | {:8.1}   | {:8.1}   | {:4.0}% | {:8.1}   | {:8.1}   | {:4.0}% | {:8.1}   | {:8.1}    | {:4.0}%",
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

fn warmup_gpu(queue: &metal::CommandQueueRef, pipelines: &[&metal::ComputePipelineState]) {
    // Create a small buffer for warmup
    let warmup_size = 4096;
    let input = vec![1.0f32; warmup_size];
    let input_buffer = queue.device().new_buffer_with_data(
        input.as_ptr() as *const _,
        (warmup_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = queue.device().new_buffer(
        (warmup_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Run each pipeline 50 times to warm up GPU scheduler
    for _ in 0..50 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        for pipeline in pipelines {
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            let grid = MTLSize { width: warmup_size as u64, height: 1, depth: 1 };
            let tg = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_threads(grid, tg);
        }
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }
}

fn create_pipeline(device: &Device, library: &metal::LibraryRef, name: &str) -> metal::ComputePipelineState {
    let kernel = library.get_function(name, None).expect(&format!("Failed to get {}", name));
    device.new_compute_pipeline_state_with_function(&kernel).expect(&format!("Failed to create pipeline for {}", name))
}

/// Run benchmark alternating between our and llama kernels
/// Returns median time for the specified pipeline (return_our=true for our, false for llama)
fn run_benchmark(
    queue: &metal::CommandQueueRef,
    our_pipeline: &metal::ComputePipelineState,
    llama_pipeline: &metal::ComputePipelineState,
    input: &metal::BufferRef,
    output: &metal::BufferRef,
    n: usize,
    dispatches_per_run: usize,
    runs: usize,
    return_our: bool,
    use_float4: bool,
) -> f64 {
    const DISPATCH_THRESHOLD: usize = 32768;
    let ne00 = if use_float4 { (n / 4) as i32 } else { n as i32 };

    // Determine dispatch mode for our kernel based on size
    let use_dispatch_threads = n < DISPATCH_THRESHOLD;

    // Our dispatch parameters
    let (our_tg, our_grid, our_is_tg) = if use_dispatch_threads {
        // dispatch_threads mode
        (MTLSize { width: 256, height: 1, depth: 1 }, MTLSize { width: n as u64, height: 1, depth: 1 }, false)
    } else {
        // dispatch_thread_groups mode
        let n_elements = if use_float4 { n / 4 } else { n };
        (MTLSize { width: 256, height: 1, depth: 1 }, MTLSize { width: ((n_elements + 255) / 256) as u64, height: 1, depth: 1 }, true)
    };

    // llama.cpp dispatch: depends on size threshold (32768)
    let (llama_tg, llama_grid, llama_is_cnt) = if n < DISPATCH_THRESHOLD {
        // cnt mode: 1 thread per group
        (MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: n as u64, height: 1, depth: 1 }, true)
    } else {
        // non-cnt mode: 256 threads per group
        let n_elements = if use_float4 { n / 4 } else { n };
        (MTLSize { width: 256, height: 1, depth: 1 }, MTLSize { width: ((n_elements + 255) / 256) as u64, height: 1, depth: 1 }, false)
    };

    let mut our_times = Vec::with_capacity(runs);
    let mut llama_times = Vec::with_capacity(runs);

    for _ in 0..runs {
        // Run llama.cpp pipeline FIRST (swapped order)
        let start2 = Instant::now();
        let cmd_buf2 = queue.new_command_buffer();
        let encoder2 = cmd_buf2.new_compute_command_encoder();
        encoder2.set_compute_pipeline_state(llama_pipeline);
        encoder2.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder2.set_buffer(1, Some(input), 0);
        encoder2.set_buffer(2, Some(output), 0);
        if llama_is_cnt {
            // cnt mode: simple dispatch with 1 thread per group
            for _ in 0..dispatches_per_run {
                encoder2.dispatch_thread_groups(llama_grid, llama_tg);
            }
        } else {
            // non-cnt mode: dispatch with 256 threads per group
            for _ in 0..dispatches_per_run {
                encoder2.dispatch_thread_groups(llama_grid, llama_tg);
            }
        }
        encoder2.end_encoding();
        cmd_buf2.commit();
        cmd_buf2.wait_until_completed();
        llama_times.push(start2.elapsed().as_secs_f64() / dispatches_per_run as f64);

        // Run our pipeline SECOND
        let start1 = Instant::now();
        let cmd_buf1 = queue.new_command_buffer();
        let encoder1 = cmd_buf1.new_compute_command_encoder();
        encoder1.set_compute_pipeline_state(our_pipeline);
        if our_is_tg {
            encoder1.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
            encoder1.set_buffer(1, Some(input), 0);
            encoder1.set_buffer(2, Some(output), 0);
            for _ in 0..dispatches_per_run {
                encoder1.dispatch_thread_groups(our_grid, our_tg);
            }
        } else {
            // dispatch_threads mode doesn't need ne00
            encoder1.set_buffer(0, Some(input), 0);
            encoder1.set_buffer(1, Some(output), 0);
            for _ in 0..dispatches_per_run {
                encoder1.dispatch_threads(our_grid, our_tg);
            }
        }
        encoder1.end_encoding();
        cmd_buf1.commit();
        cmd_buf1.wait_until_completed();
        our_times.push(start1.elapsed().as_secs_f64() / dispatches_per_run as f64);
    }

    // Return median of the requested pipeline
    let times = if return_our { &our_times } else { &llama_times };
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[runs / 2]
}
