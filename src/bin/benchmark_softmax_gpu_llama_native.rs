// Benchmark: Our GPU Softmax vs llama.cpp NATIVE Metal kernel
// Directly loads llama.cpp's softmax.metal without modification
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

impl OurSoftmaxMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("../../shaders/softmax.metal");
        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library.get_function("kernel_soft_max_f32_fast", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let kernel_vec4 = library.get_function("kernel_soft_max_f32_4_fast", None)
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

    fn softmax(&self, src: &[f32], n_heads: usize, seq_len: usize, _scale: f32, iterations: usize) -> f64 {
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
        let ne01 = n_heads as i32;

        // Warmup
        for _ in 0..100 {
            let _ = self.dispatch(&input_buffer, &output_buffer, ne00, ne01, n_heads, batch_size, threads_per_row, pipeline, shmem_size);
        }

        // Measure using batched dispatch
        let batch_iters = 100;
        let num_batches = iterations / batch_iters;
        let mut total_time = 0.0f64;

        for _ in 0..num_batches {
            let start = std::time::Instant::now();
            {
                let command_buffer = self.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
                encoder.set_buffer(1, Some(&input_buffer), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.set_bytes(3, 4, &ne01 as *const i32 as *const std::ffi::c_void);

                let grid_size = MTLSize { width: n_heads as u64, height: batch_size as u64, depth: 1 };
                let threadgroup_size = MTLSize { width: threads_per_row as u64, height: 1, depth: 1 };
                encoder.set_threadgroup_memory_length(0, shmem_size as u64);

                for _ in 0..batch_iters {
                    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
                }
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
            total_time += start.elapsed().as_secs_f64();
        }

        total_time / (num_batches * batch_iters) as f64
    }

    fn dispatch(&self, input: &Buffer, output: &Buffer, ne00: i32, ne01: i32, n_heads: usize, batch_size: usize, threads_per_row: usize, pipeline: &ComputePipelineState, shmem_size: usize) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &ne01 as *const i32 as *const std::ffi::c_void);

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

// llama.cpp NATIVE kernel context - loads exact llama.cpp softmax.metal
struct LlamaNativeSoftmaxContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pipeline_vec4: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

// Exact match of ggml_metal_kargs_soft_max from llama.cpp
#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlMetalKargsSoftMax {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
}

impl LlamaNativeSoftmaxContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        // Load llama.cpp's exact softmax.metal kernel
        let source = r#"
#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define N_SIMDWIDTH 32

struct ggml_metal_kargs_soft_max {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
};

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;
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

        // Use same thread configuration as our kernel for fair comparison
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let nth = seq_len.min(max_threads).next_power_of_two();

        let simd_groups = (nth + 31) / 32;
        let shmem_size = simd_groups * 4;

        // Build args struct matching llama.cpp exactly
        let args = GgmlMetalKargsSoftMax {
            ne00: seq_len as i32,
            ne01: n_heads as i32,
            ne02: batch_size as i32,
            nb01: (seq_len * 4) as u64,
            nb02: (n_heads * seq_len * 4) as u64,
            nb03: 0,
            ne11: 0,
            ne12: n_heads as i32,
            ne13: batch_size as i32,
            nb11: 0,
            nb12: (n_heads * seq_len * 4) as u64,
            nb13: 0,
            nb1: (seq_len * 4) as u64,
            nb2: (n_heads * seq_len * 4) as u64,
            nb3: 0,
            scale,
            max_bias: 0.0,
            m0: 0.0,
            m1: 0.0,
            n_head_log2: 0,
        };

        // Warmup
        for _ in 0..100 {
            let _ = self.dispatch(&input_buffer, &output_buffer, &args, n_heads, batch_size, nth, pipeline, shmem_size);
        }

        // Measure using batched dispatch
        let batch_iters = 100;
        let num_batches = iterations / batch_iters;
        let mut total_time = 0.0f64;

        for _ in 0..num_batches {
            let start = std::time::Instant::now();
            {
                let command_buffer = self.queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_bytes(0, std::mem::size_of::<GgmlMetalKargsSoftMax>() as u64, &args as *const GgmlMetalKargsSoftMax as *const std::ffi::c_void);
                encoder.set_buffer(1, Some(&input_buffer), 0);
                // src1 (mask) - use src0 as placeholder (kernel checks src1 != src0)
                encoder.set_buffer(2, Some(&input_buffer), 0);
                // src2 (psrc2) - use src0 as placeholder (kernel checks src2 != src0)
                encoder.set_buffer(3, Some(&input_buffer), 0);
                encoder.set_buffer(4, Some(&output_buffer), 0);

                let grid_size = MTLSize { width: n_heads as u64, height: batch_size as u64, depth: 1 };
                let threadgroup_size = MTLSize { width: nth as u64, height: 1, depth: 1 };
                encoder.set_threadgroup_memory_length(0, shmem_size as u64);

                for _ in 0..batch_iters {
                    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
                }
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
            total_time += start.elapsed().as_secs_f64();
        }

        total_time / (num_batches * batch_iters) as f64
    }

    fn dispatch(&self, input: &Buffer, output: &Buffer, args: &GgmlMetalKargsSoftMax, n_heads: usize, batch_size: usize, nth: usize, pipeline: &ComputePipelineState, shmem_size: usize) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GgmlMetalKargsSoftMax>() as u64, args as *const GgmlMetalKargsSoftMax as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(input), 0);
        encoder.set_buffer(3, Some(input), 0);
        encoder.set_buffer(4, Some(output), 0);

        let grid_size = MTLSize { width: n_heads as u64, height: batch_size as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: nth as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, shmem_size as u64);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

fn main() {
    println!("=== GPU Softmax: Our Implementation vs llama.cpp Native Kernel ===\n");

    let ours = OurSoftmaxMetalContext::new().expect("Failed to create our context");
    let llama = LlamaNativeSoftmaxContext::new().expect("Failed to create llama.cpp context");

    let scale = 1.0f32;
    let iterations = 1000;

    // Test configs: (n_heads, seq_len)
    let configs: [(usize, usize); 7] = [
        (32, 64),
        (32, 128),
        (32, 256),
        (32, 512),
        (32, 1024),
        (32, 2048),
        (32, 4096),
    ];

    println!("Config      | Our (μs) | llama.cpp (μs) | Ratio");
    println!("------------|----------|----------------|-------");

    for (n_heads, seq_len) in configs {
        let total_size = n_heads * seq_len;
        let src: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.01).sin()).collect();

        let t_ours = ours.softmax(&src, n_heads, seq_len, scale, iterations) * 1e6;
        let t_llama = llama.softmax(&src, n_heads, seq_len, scale, iterations) * 1e6;

        let ratio = t_ours / t_llama * 100.0;
        println!("{:2}h{:4}s  | {:8.2} | {:14.2} | {:5.1}%",
            n_heads, seq_len, t_ours, t_llama, ratio);
    }

    println!("\n=== Interpretation ===");
    println!("Ratio > 100%: Our kernel is FASTER than llama.cpp");
    println!("Ratio < 100%: Our kernel is SLOWER - need to investigate");
}
