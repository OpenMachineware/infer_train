use metal::{Device, CommandQueue, Library, ComputePipelineState, MTLSize, Buffer};
use std::cell::RefCell;

// Our Metal context with both scalar and vec4 kernels
struct RmsNormMetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,
    pipeline: ComputePipelineState,
    pipeline_vec4: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    weight_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    weight_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

impl RmsNormMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("../../shaders/rms_norm.metal");
        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library.get_function("kernel_rms_norm_f32", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let kernel_vec4 = library.get_function("kernel_rms_norm_f32_4", None)
            .map_err(|e| format!("Failed to get vec4 kernel: {}", e))?;
        let pipeline_vec4 = device.new_compute_pipeline_state_with_function(&kernel_vec4)
            .map_err(|e| format!("Failed to create vec4 pipeline: {}", e))?;

        Ok(Self {
            device, queue, library, pipeline, pipeline_vec4,
            input_buffer: RefCell::new(None),
            weight_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            weight_size: RefCell::new(0),
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

    fn rms_norm(&self, x: &[f32], w: &[f32], hidden_dim: usize, eps: f32, iterations: usize) -> f64 {
        let n_rows = x.len() / hidden_dim;

        let input_buffer = self.get_buffer(x.len() * 4, &self.input_buffer, &self.input_size);
        let weight_buffer = self.get_buffer(w.len() * 4, &self.weight_buffer, &self.weight_size);
        let output_buffer = self.get_buffer(x.len() * 4, &self.output_buffer, &self.output_size);

        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), input_buffer.contents() as *mut f32, x.len());
            std::ptr::copy_nonoverlapping(w.as_ptr(), weight_buffer.contents() as *mut f32, w.len());
        }

        let use_vec4 = hidden_dim % 4 == 0;
        let pipeline = if use_vec4 { &self.pipeline_vec4 } else { &self.pipeline };
        let ne00 = hidden_dim as i32;
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;

        // Warm up GPU
        for _ in 0..20 {
            let _ = self.dispatch(&input_buffer, &weight_buffer, &output_buffer, ne00, n_rows, eps, max_threads, pipeline);
        }

        // Measure individual dispatches to find min (peak performance)
        let mut min_time = f64::MAX;
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.dispatch(&input_buffer, &weight_buffer, &output_buffer, ne00, n_rows, eps, max_threads, pipeline);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed < min_time {
                min_time = elapsed;
            }
        }

        min_time
    }

    fn dispatch(&self, input: &Buffer, weight: &Buffer, output: &Buffer, ne00: i32, n_rows: usize, eps: f32, max_threads: usize, pipeline: &ComputePipelineState) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);

        // Key fix: for vec4 kernel, use ne00/4 for thread calculation (like llama.cpp)
        let use_vec4 = ne00 % 4 == 0;
        let ne00_t = if use_vec4 { ne00 / 4 } else { ne00 };

        let mut nth = 32;
        while nth < ne00_t as usize && nth < max_threads {
            nth *= 2;
        }
        nth = nth.min(max_threads).min((ne00_t as usize + 31) / 32 * 32);

        let grid_size = MTLSize { width: n_rows as u64, height: 1, depth: 1 };
        let threadgroup_size = MTLSize { width: nth as u64, height: 1, depth: 1 };

        let simd_groups = (nth + 31) / 32;
        let shmem_size = std::cmp::max(32, simd_groups) * 4;
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

// llama.cpp-style kernel
struct LlamaCppMetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,
    pipeline: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    weight_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    weight_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

impl LlamaCppMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = r#"
#include <metal_stdlib>
using namespace metal;

struct rms_norm_args {
    int32_t ne00;
    int32_t ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float eps;
    int32_t nef1[4];
    int32_t nef2[4];
    int32_t nef3[4];
    uint64_t nbf1[4];
    uint64_t nbf2[4];
    uint64_t nbf3[4];
    float scale;
    uint64_t nb3_out;
    uint64_t nb2_out;
    uint64_t nb1_out;
};

kernel void kernel_rms_norm_f32(
        constant rms_norm_args & args,
        device const char * src0,
        device const char * src1_0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;

    device const float * x = (device const float *) (src0 + i01*args.nb1);
    device const float * w = (device const float *) src1_0;

    float sumf = 0.0f;

    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += x[i00] * x[i00];
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf / args.ne00;
    const float scale = 1.0f / sqrt(mean + args.eps);

    device float * y = (device float *) (dst + i01*args.nb1_out);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        y[i00] = x[i00] * scale * w[i00];
    }
}
"#;

        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library.get_function("kernel_rms_norm_f32", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        Ok(Self {
            device, queue, library, pipeline,
            input_buffer: RefCell::new(None),
            weight_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            weight_size: RefCell::new(0),
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

    fn rms_norm(&self, x: &[f32], w: &[f32], hidden_dim: usize, eps: f32, iterations: usize) -> f64 {
        let n_rows = x.len() / hidden_dim;
        let row_bytes = hidden_dim * 4;

        let input_buffer = self.get_buffer(x.len() * 4, &self.input_buffer, &self.input_size);
        let weight_buffer = self.get_buffer(w.len() * 4, &self.weight_buffer, &self.weight_size);
        let output_buffer = self.get_buffer(x.len() * 4, &self.output_buffer, &self.output_size);

        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), input_buffer.contents() as *mut f32, x.len());
            std::ptr::copy_nonoverlapping(w.as_ptr(), weight_buffer.contents() as *mut f32, w.len());
        }

        let max_threads = self.pipeline.max_total_threads_per_threadgroup() as usize;

        #[repr(C)]
        struct RmsNormArgs {
            ne00: i32, ne00_t: i32, nb1: u64, nb2: u64, nb3: u64, eps: f32,
            nef1: [i32; 4], nef2: [i32; 4], nef3: [i32; 4],
            nbf1: [u64; 4], nbf2: [u64; 4], nbf3: [u64; 4],
            scale: f32, nb3_out: u64, nb2_out: u64, nb1_out: u64,
        }

        let args = RmsNormArgs {
            ne00: hidden_dim as i32, ne00_t: hidden_dim as i32,
            nb1: row_bytes as u64, nb2: 0, nb3: 0, eps,
            nef1: [hidden_dim as i32, 0, 0, 0], nef2: [1, 0, 0, 0], nef3: [1, 0, 0, 0],
            nbf1: [row_bytes as u64, 0, 0, 0], nbf2: [0, 0, 0, 0], nbf3: [0, 0, 0, 0],
            scale: 1.0, nb3_out: 0, nb2_out: 0, nb1_out: row_bytes as u64,
        };

        let args_bytes = unsafe {
            std::slice::from_raw_parts(&args as *const RmsNormArgs as *const u8, std::mem::size_of::<RmsNormArgs>())
        };

        for _ in 0..20 {
            let _ = self.dispatch(&input_buffer, &weight_buffer, &output_buffer, args_bytes, n_rows, max_threads);
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = self.dispatch(&input_buffer, &weight_buffer, &output_buffer, args_bytes, n_rows, max_threads);
        }
        let elapsed = start.elapsed();

        elapsed.as_secs_f64() / iterations as f64
    }

    fn dispatch(&self, input: &Buffer, weight: &Buffer, output: &Buffer, args: &[u8], n_rows: usize, max_threads: usize) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_bytes(0, args.len() as u64, args.as_ptr() as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(output), 0);

        let ne00 = unsafe { *(args.as_ptr() as *const i32) };
        let mut nth = 32;
        while nth < ne00 as usize && nth < max_threads {
            nth *= 2;
        }
        nth = nth.min(max_threads).min((ne00 as usize + 31) / 32 * 32);

        let grid_size = MTLSize { width: n_rows as u64, height: 1, depth: 1 };
        let threadgroup_size = MTLSize { width: nth as u64, height: 1, depth: 1 };

        let simd_groups = (nth + 31) / 32;
        let shmem_size = std::cmp::max(32, simd_groups) * 4;
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

fn main() {
    let ours = RmsNormMetalContext::new().expect("Failed to create our context");
    let llama = LlamaCppMetalContext::new().expect("Failed to create llama.cpp context");

    println!("=== Metal RMSNorm: Our Implementation vs llama.cpp ===\n");

    let hidden_dims = vec![512, 1024, 2048, 4096, 8192];
    let batch_sizes = vec![1, 4, 16, 64, 256];
    let eps = 1e-5f32;
    let iterations = 5000;  // More iterations for better accuracy

    println!("Hidden Dimension Scaling (batch=1, {} iter, 5 warmup):", iterations);
    println!("{:<10} {:>12} {:>12} {:>7}", "Dim", "Ours (us)", "llama (us)", "Perf");
    println!("{}", "-".repeat(50));

    for hidden_dim in &hidden_dims {
        let x: Vec<f32> = (0..*hidden_dim).map(|i| ((i % 100) as f32 + 1.0) / 50.0).collect();
        let w: Vec<f32> = vec![1.0; *hidden_dim];

        let t_ours = ours.rms_norm(&x, &w, *hidden_dim, eps, iterations) * 1e6;
        let t_llama = llama.rms_norm(&x, &w, *hidden_dim, eps, iterations) * 1e6;
        let ratio = t_llama / t_ours;  // llama/ours: >1.0 means we're faster

        println!("{:<10} {:>12.2} {:>12.2} {:>7.1}%", hidden_dim, t_ours, t_llama, ratio * 100.0);
    }

    println!("\nBatch Size Scaling (hidden_dim=4096, {} iter, 5 warmup):", iterations);
    println!("{:<10} {:>12} {:>12} {:>7}", "Batch", "Ours (us)", "llama (us)", "Perf");
    println!("{}", "-".repeat(50));

    let hidden_dim = 4096;
    for batch in &batch_sizes {
        let x: Vec<f32> = (0..hidden_dim * batch).map(|i| ((i % 100) as f32 + 1.0) / 50.0).collect();
        let w: Vec<f32> = vec![1.0; hidden_dim];

        let t_ours = ours.rms_norm(&x, &w, hidden_dim, eps, iterations) * 1e6;
        let t_llama = llama.rms_norm(&x, &w, hidden_dim, eps, iterations) * 1e6;
        let ratio = t_llama / t_ours;  // llama/ours: >1.0 means we're faster

        println!("{:<10} {:>12.2} {:>12.2} {:>7.1}%", batch, t_ours, t_llama, ratio * 100.0);
    }
}
