// Benchmark Softmax using alternating A-B-A-B methodology
// Sequential testing has FALSE gaps due to GPU frequency variation
// MUST alternate between implementations for fair comparison

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLSize};
use std::cell::RefCell;

struct SoftmaxMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pipeline_vec4: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

impl SoftmaxMetalContext {
    fn new(shader_source: &str, kernel_name: &str) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = format!(
            r#"
#include <metal_stdlib>
using namespace metal;

constant float N_INFINITY = 1e30f;

{}
"#,
            shader_source
        );

        let library = device
            .new_library_with_source(&source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library
            .get_function(kernel_name, None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let kernel_vec4_name = format!("{}_4", kernel_name);
        let kernel_vec4 = library.get_function(&kernel_vec4_name, None).ok();
        let pipeline_vec4 = if let Some(k) = kernel_vec4 {
            device
                .new_compute_pipeline_state_with_function(&k)
                .map_err(|e| format!("Failed to create vec4 pipeline: {}", e))?
        } else {
            pipeline.clone()
        };

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_vec4,
            input_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            output_size: RefCell::new(0),
        })
    }

    fn get_buffer(
        &self,
        size: usize,
        buffer_ref: &RefCell<Option<Buffer>>,
        size_ref: &RefCell<usize>,
    ) -> Buffer {
        let mut buffer_cell = buffer_ref.borrow_mut();
        let mut size_cell = size_ref.borrow_mut();
        if *size_cell != size {
            *buffer_cell = Some(self.device.new_buffer(
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            *size_cell = size;
        }
        buffer_cell.as_ref().unwrap().clone()
    }

    fn softmax(
        &self,
        src: &[f32],
        n_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> f64 {
        let batch_size = src.len() / (n_heads * seq_len);
        let total_size = src.len();

        let input_buffer = self.get_buffer(
            total_size * 4,
            &self.input_buffer,
            &self.input_size,
        );
        let output_buffer = self.get_buffer(
            total_size * 4,
            &self.output_buffer,
            &self.output_size,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                input_buffer.contents() as *mut f32,
                total_size,
            );
        }

        let use_vec4 = seq_len % 4 == 0;
        let pipeline =
            if use_vec4 { &self.pipeline_vec4 } else { &self.pipeline };
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_row = seq_len.min(max_threads).next_power_of_two();
        let simd_groups = (threads_per_row + 31) / 32;
        let shmem_size = simd_groups * 4;
        let ne00 = seq_len as i32;
        let ne01 = n_heads as i32;

        // Single dispatch for timing
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &ne01 as *const i32 as *const std::ffi::c_void);

        let grid_size = MTLSize {
            width: n_heads as u64,
            height: batch_size as u64,
            depth: 1,
        };
        let threadgroup_size =
            MTLSize { width: threads_per_row as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, shmem_size as u64);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let start = command_buffer.gpu_start_time().unwrap();
        let end = command_buffer.gpu_end_time().unwrap();
        end - start
    }
}

fn main() {
    println!(
        "=== Metal Softmax: Alternating Benchmark (Fair Comparison) ===\n"
    );
    println!("Methodology: A-B-A-B alternating, 500 rounds, eliminates GPU frequency variance\n");

    // Same kernel code for both
    let kernel_code = r#"
kernel void kernel_soft_max_f32(
    constant int32_t & ne00 [[buffer(0)]],
    device const float * src [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant int32_t & ne01 [[buffer(3)]],
    threadgroup float * buf [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const float N_INFINITY = 1e30f;

    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;
    const uint64_t nb01 = ne00 * 4;
    const uint64_t nb02 = ne01 * nb01;

    const uint64_t row_offset = i02 * nb02 + i01 * nb01;
    device const float * psrc = src + row_offset / 4;
    device float * pdst = dst + row_offset / 4;

    float lmax = -N_INFINITY;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        lmax = max(lmax, psrc[i00]);
    }

    float max_val = simd_max(lmax);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = -N_INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = max_val; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        const float exp_val = exp(psrc[i00] - max_val);
        lsum += exp_val;
        pdst[i00] = exp_val;
    }

    float sum = simd_sum(lsum);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    threadgroup_barrier(mem_flags::mem_none);

    const float inv_sum = 1.0f / sum;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}
"#;

    let ours = SoftmaxMetalContext::new(kernel_code, "kernel_soft_max_f32")
        .expect("Failed to create our context");
    let llama = SoftmaxMetalContext::new(kernel_code, "kernel_soft_max_f32")
        .expect("Failed to create llama context");

    // Test configurations
    let configs: Vec<(usize, usize)> = vec![
        (32, 64),
        (32, 128),
        (32, 256),
        (32, 512),
        (32, 1024),
        (32, 2048),
        (32, 4096),
    ];

    let batch_size = 4;
    let rounds = 500;

    println!("Batch size = {}", batch_size);
    println!("Rounds per config = {}", rounds);
    println!(
        "{:<15} {:>12} {:>12} {:>7}",
        "Config", "Ours (us)", "llama (us)", "Perf"
    );
    println!("{}", "-".repeat(55));

    for (n_heads, seq_len) in &configs {
        let total_size = batch_size * n_heads * seq_len;
        let src: Vec<f32> =
            (0..total_size).map(|i| ((i % 100) as f32) / 10.0).collect();

        // Alternating test: A-B-A-B-...
        let mut times_ours = Vec::new();
        let mut times_llama = Vec::new();

        for _ in 0..rounds {
            times_ours.push(ours.softmax(&src, *n_heads, *seq_len, 1.0));
            times_llama.push(llama.softmax(&src, *n_heads, *seq_len, 1.0));
        }

        let avg_ours: f64 = times_ours.iter().sum::<f64>() / rounds as f64;
        let avg_llama: f64 = times_llama.iter().sum::<f64>() / rounds as f64;

        let t_ours = avg_ours * 1e6;
        let t_llama = avg_llama * 1e6;
        let ratio = t_llama / t_ours;

        println!(
            "{:<15} {:>12.2} {:>12.2} {:>6.1}%",
            format!("{}h{}s", n_heads, seq_len),
            t_ours,
            t_llama,
            ratio * 100.0
        );
    }

    println!("\n=== Note ===");
    println!("Both use identical kernel code and compilation flags.");
    println!(
        "Performance should be exactly the same (within thermal variance)."
    );
}
