// Benchmark GPU Softmax vs llama.cpp Metal implementation
use metal::{Device, CommandQueue, ComputePipelineState, MTLSize, Buffer};
use std::cell::RefCell;

// Our Metal Softmax context
struct OurSoftmaxMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pipeline_vec4: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

#[repr(C)]
struct SoftmaxArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    scale: f32,
}

impl OurSoftmaxMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("../../shaders/softmax.metal");
        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library.get_function("kernel_soft_max_f32", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let kernel_vec4 = library.get_function("kernel_soft_max_f32_4", None)
            .map_err(|e| format!("Failed to get vec4 kernel: {}", e))?;
        let pipeline_vec4 = device.new_compute_pipeline_state_with_function(&kernel_vec4)
            .map_err(|e| format!("Failed to create vec4 pipeline: {}", e))?;

        Ok(Self {
            device, queue, pipeline, pipeline_vec4,
            input_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            output_size: RefCell::new(0),
        })
    }

    fn get_buffer(&self, size: usize, buffer_ref: &RefCell<Option<Buffer>>, size_ref: &RefCell<usize>) -> Buffer {
        let mut buffer_cell = buffer_ref.borrow_mut();
        let mut size_cell = size_ref.borrow_mut();
        if *size_cell != size {
            *buffer_cell = Some(self.device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared));
            *size_cell = size;
        }
        buffer_cell.as_ref().unwrap().clone()
    }

    fn softmax(&self, src: &[f32], n_heads: usize, seq_len: usize, scale: f32, iterations: usize) -> f64 {
        let batch_size = src.len() / (n_heads * seq_len);
        let total_size = src.len();

        let input_buffer = self.get_buffer(total_size * 4, &self.input_buffer, &self.input_size);
        let output_buffer = self.get_buffer(total_size * 4, &self.output_buffer, &self.output_size);

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), input_buffer.contents() as *mut f32, total_size);
        }

        let args = SoftmaxArgs {
            ne00: seq_len as i32,
            ne01: n_heads as i32,
            ne02: batch_size as i32,
            nb01: (seq_len * 4) as u64,
            nb02: (n_heads * seq_len * 4) as u64,
            nb03: 0,
            scale,
        };

        let use_vec4 = seq_len % 4 == 0;
        let pipeline = if use_vec4 { &self.pipeline_vec4 } else { &self.pipeline };
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_row = seq_len.min(max_threads).next_power_of_two();
        let simd_groups = (threads_per_row + 31) / 32;
        let shmem_size = simd_groups * 4;

        // Warmup
        for _ in 0..20 {
            let _ = self.dispatch(&input_buffer, &output_buffer, &args, n_heads, batch_size, threads_per_row, pipeline, shmem_size);
        }

        // Measure
        let mut min_time = f64::MAX;
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.dispatch(&input_buffer, &output_buffer, &args, n_heads, batch_size, threads_per_row, pipeline, shmem_size);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed < min_time {
                min_time = elapsed;
            }
        }

        min_time
    }

    fn dispatch(&self, input: &Buffer, output: &Buffer, args: &SoftmaxArgs, n_heads: usize, batch_size: usize, threads_per_row: usize, pipeline: &ComputePipelineState, shmem_size: usize) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, std::mem::size_of::<SoftmaxArgs>() as u64, args as *const SoftmaxArgs as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);

        let grid_size = MTLSize { width: n_heads as u64, height: batch_size as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: threads_per_row as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, shmem_size as u64);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

// llama.cpp-style Softmax context
struct LlamaCppSoftmaxMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pipeline_vec4: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

#[repr(C)]
struct LlamaSoftmaxArgs {
    ne00: i32, ne01: i32, ne02: i32,
    nb01: u64, nb02: u64, nb03: u64,
    ne11: i32, ne12: i32, ne13: i32,
    nb11: u64, nb12: u64, nb13: u64,
    nb1: u64, nb2: u64, nb3: u64,
    scale: f32, max_bias: f32, m0: f32, m1: f32, n_head_log2: i32,
}

impl LlamaCppSoftmaxMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kernel_soft_max_f32(
    constant int32_t & ne00 [[buffer(0)]],
    device const float * src [[buffer(1)]],
    device float * dst [[buffer(2)]],
    threadgroup float * buf [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const float scale = 1.0f;
    const float N_INFINITY = 1e30f;

    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;
    const uint64_t nb01 = ne00 * 4;
    const uint64_t nb02 = nb01;

    const uint64_t row_offset = i02 * nb02 + i01 * nb01;
    device const float* psrc = src + row_offset / 4;
    device float* pdst = dst + row_offset / 4;

    float lmax = -N_INFINITY;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        lmax = max(lmax, psrc[i00] * scale);
    }

    float max_val = simd_max(lmax);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = -N_INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = max_val; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = simd_max(buf[tiisg]);
    }

    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        const float exp_val = exp((psrc[i00] * scale) - max_val);
        lsum += exp_val;
        pdst[i00] = exp_val;
    }

    float sum = simd_sum(lsum);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(buf[tiisg]);
    }

    const float inv_sum = 1.0f / sum;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

kernel void kernel_soft_max_f32_4(
    constant int32_t & ne00 [[buffer(0)]],
    device const float * src [[buffer(1)]],
    device float * dst [[buffer(2)]],
    threadgroup float * buf [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const float scale = 1.0f;
    const float N_INFINITY = 1e30f;

    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;
    const uint64_t nb01 = ne00 * 4;
    const uint64_t nb02 = nb01;

    const uint64_t row_offset = i02 * nb02 + i01 * nb01;
    device const float4* psrc4 = (device const float4*)(src + row_offset / 4);
    device float4* pdst4 = (device float4*)(dst + row_offset / 4);
    const int32_t ne00_4 = ne00 / 4;

    float4 lmax4 = -N_INFINITY;
    for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00] * scale);
    }
    float lmax = max(max(lmax4[0], lmax4[1]), max(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = -N_INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = max_val; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = simd_max(buf[tiisg]);
    }

    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
        const float4 exp_val = exp((psrc4[i00] * scale) - max_val);
        lsum4 += exp_val;
        pdst4[i00] = exp_val;
    }
    float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    float sum = simd_sum(lsum);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(buf[tiisg]);
    }

    const float inv_sum = 1.0f / sum;
    for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}
"#;

        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library.get_function("kernel_soft_max_f32", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let kernel_vec4 = library.get_function("kernel_soft_max_f32_4", None)
            .map_err(|e| format!("Failed to get vec4 kernel: {}", e))?;
        let pipeline_vec4 = device.new_compute_pipeline_state_with_function(&kernel_vec4)
            .map_err(|e| format!("Failed to create vec4 pipeline: {}", e))?;

        Ok(Self {
            device, queue, pipeline, pipeline_vec4,
            input_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            output_size: RefCell::new(0),
        })
    }

    fn get_buffer(&self, size: usize, buffer_ref: &RefCell<Option<Buffer>>, size_ref: &RefCell<usize>) -> Buffer {
        let mut buffer_cell = buffer_ref.borrow_mut();
        let mut size_cell = size_ref.borrow_mut();
        if *size_cell != size {
            *buffer_cell = Some(self.device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared));
            *size_cell = size;
        }
        buffer_cell.as_ref().unwrap().clone()
    }

    fn softmax(&self, src: &[f32], n_heads: usize, seq_len: usize, scale: f32, iterations: usize) -> f64 {
        let batch_size = src.len() / (n_heads * seq_len);
        let total_size = src.len();

        let input_buffer = self.get_buffer(total_size * 4, &self.input_buffer, &self.input_size);
        let output_buffer = self.get_buffer(total_size * 4, &self.output_buffer, &self.output_size);

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), input_buffer.contents() as *mut f32, total_size);
        }

        let use_vec4 = seq_len % 4 == 0;
        let pipeline = if use_vec4 { &self.pipeline_vec4 } else { &self.pipeline };
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_row = seq_len.min(max_threads).next_power_of_two();
        let simd_groups = (threads_per_row + 31) / 32;
        let shmem_size = simd_groups * 4;

        let ne00 = seq_len as i32;

        // Warmup
        for _ in 0..20 {
            let _ = self.dispatch(&input_buffer, &output_buffer, ne00, n_heads, batch_size, threads_per_row, pipeline, shmem_size);
        }

        // Measure
        let mut min_time = f64::MAX;
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.dispatch(&input_buffer, &output_buffer, ne00, n_heads, batch_size, threads_per_row, pipeline, shmem_size);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed < min_time {
                min_time = elapsed;
            }
        }

        min_time
    }

    fn dispatch(&self, input: &Buffer, output: &Buffer, ne00: i32, n_heads: usize, batch_size: usize, threads_per_row: usize, pipeline: &ComputePipelineState, shmem_size: usize) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);

        let grid_size = MTLSize { width: n_heads as u64, height: batch_size as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: threads_per_row as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, shmem_size as u64);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

fn main() {
    let ours = OurSoftmaxMetalContext::new().expect("Failed to create our context");
    let llama = LlamaCppSoftmaxMetalContext::new().expect("Failed to create llama.cpp context");

    println!("=== Metal Softmax: Our Implementation vs llama.cpp ===\n");

    // Typical attention sizes
    let configs: Vec<(usize, usize)> = vec![
        (32, 64),    // 32 heads, 64 seq
        (32, 128),   // LLaMA style
        (32, 256),
        (32, 512),
        (32, 1024),
        (32, 2048),
        (32, 4096),
    ];

    let batch_size = 4;
    let scale = 1.0f32;
    let iterations = 5000;

    println!("Batch size = {}", batch_size);
    println!("{:<15} {:>12} {:>12} {:>7}", "Config", "Ours (us)", "llama (us)", "Perf");
    println!("{}", "-".repeat(55));

    for (n_heads, seq_len) in &configs {
        let total_size = batch_size * n_heads * seq_len;
        let src: Vec<f32> = (0..total_size).map(|i| ((i % 100) as f32) / 10.0).collect();

        let t_ours = ours.softmax(&src, *n_heads, *seq_len, scale, iterations) * 1e6;
        let t_llama = llama.softmax(&src, *n_heads, *seq_len, scale, iterations) * 1e6;
        let ratio = t_llama / t_ours;

        println!("{:<15} {:>12.2} {:>12.2} {:>7.1}%",
            format!("{}h{}s", n_heads, seq_len), t_ours, t_llama, ratio * 100.0);
    }

    // Correctness check
    println!("\n=== Correctness ===");
    let n_heads = 32;
    let seq_len = 128;
    let total_size = batch_size * n_heads * seq_len;
    let src: Vec<f32> = (0..total_size).map(|i| ((i % 100) as f32) / 10.0).collect();

    // Get outputs from both implementations
    let input_buffer = ours.get_buffer(total_size * 4, &ours.input_buffer, &ours.input_size);
    let output_buffer = ours.get_buffer(total_size * 4, &ours.output_buffer, &ours.output_size);
    let args = SoftmaxArgs {
        ne00: seq_len as i32, ne01: n_heads as i32, ne02: batch_size as i32,
        nb01: (seq_len * 4) as u64, nb02: (n_heads * seq_len * 4) as u64, nb03: 0, scale,
    };

    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), input_buffer.contents() as *mut f32, total_size);
    }

    let use_vec4 = seq_len % 4 == 0;
    let pipeline = if use_vec4 { &ours.pipeline_vec4 } else { &ours.pipeline };
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    let threads_per_row = seq_len.min(max_threads).next_power_of_two();
    let simd_groups = (threads_per_row + 31) / 32;
    let shmem_size = simd_groups * 4;

    let _ = ours.dispatch(&input_buffer, &output_buffer, &args, n_heads, batch_size, threads_per_row, pipeline, shmem_size);

    let mut y_ours = vec![0.0f32; total_size];
    unsafe {
        std::ptr::copy_nonoverlapping(output_buffer.contents() as *const f32, y_ours.as_mut_ptr(), total_size);
    }

    // Verify sum = 1 for each row
    let mut max_diff = 0.0f32;
    for row in 0..(batch_size * n_heads) {
        let row_sum: f32 = y_ours[row * seq_len..(row + 1) * seq_len].iter().sum();
        let diff = (row_sum - 1.0).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max sum deviation from 1.0: {:.2e}", max_diff);
}
