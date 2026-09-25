// Benchmark GPU RoPE vs llama.cpp Metal implementation
use metal::{Device, CommandQueue, Library, ComputePipelineState, MTLSize, Buffer};
use std::cell::RefCell;

// Our Metal RoPE context
struct OurRopeMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

#[repr(C)]
struct RopeArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    n_dims: i32,
    n_offs: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_ctx_orig: i32,
}

impl OurRopeMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("../../shaders/rope.metal");
        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library.get_function("kernel_rope_neox_f32", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        Ok(Self {
            device, queue, pipeline,
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

    fn rope(&self, src: &[f32], positions: &[i32], n_heads: usize, hidden_dim: usize, n_dims: usize, freq_base: f32, iterations: usize) -> f64 {
        let n_seqs = positions.len();
        let total_size = src.len();

        let input_buffer = self.get_buffer(total_size * 4, &self.input_buffer, &self.input_size);
        let output_buffer = self.get_buffer(total_size * 4, &self.output_buffer, &self.output_size);
        let pos_buffer = self.device.new_buffer((positions.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), input_buffer.contents() as *mut f32, total_size);
            std::ptr::copy_nonoverlapping(positions.as_ptr(), pos_buffer.contents() as *mut i32, positions.len());
        }

        let args = RopeArgs {
            ne00: hidden_dim as i32,
            ne01: n_heads as i32,
            ne02: n_seqs as i32,
            nb00: 4,
            nb01: (hidden_dim * 4) as u64,
            nb02: (n_heads * hidden_dim * 4) as u64,
            n_dims: n_dims as i32,
            n_offs: 0,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig: 2048,
        };

        let max_threads = self.pipeline.max_total_threads_per_threadgroup() as usize;
        let n_dims_half = n_dims / 2;
        let threads_per_tg = n_dims_half.min(max_threads).next_power_of_two();

        // Warmup
        for _ in 0..20 {
            let _ = self.dispatch(&input_buffer, &pos_buffer, &output_buffer, &args, n_heads, n_seqs, threads_per_tg);
        }

        // Measure
        let mut min_time = f64::MAX;
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.dispatch(&input_buffer, &pos_buffer, &output_buffer, &args, n_heads, n_seqs, threads_per_tg);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed < min_time {
                min_time = elapsed;
            }
        }

        min_time
    }

    fn dispatch(&self, input: &Buffer, pos: &Buffer, output: &Buffer, args: &RopeArgs, n_heads: usize, n_seqs: usize, threads_per_tg: usize) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_bytes(0, std::mem::size_of::<RopeArgs>() as u64, args as *const RopeArgs as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(pos), 0);
        encoder.set_buffer(3, Some(output), 0);

        let grid_size = MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: threads_per_tg as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

// llama.cpp-style RoPE context
struct LlamaCppRopeMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

#[repr(C)]
struct LlamaRopeArgs {
    ne00: i32, ne01: i32, ne02: i32, ne03: i32,
    nb00: u64, nb01: u64, nb02: u64, nb03: u64,
    ne0: i32, ne1: i32, ne2: i32, ne3: i32,
    nb0: u64, nb1: u64, nb2: u64, nb3: u64,
    n_past: i32, n_dims: i32, n_offs: i32, n_ctx_orig: i32,
    freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32,
    sect_0: i32, sect_1: i32, sect_2: i32, sect_3: i32,
    src2: bool, inplace: bool,
}

impl LlamaCppRopeMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        // llama.cpp kernel_rope_neox from rope.metal
        let source = r#"
#include <metal_stdlib>
using namespace metal;

struct ggml_metal_kargs_rope {
    int32_t  ne00; int32_t  ne01; int32_t  ne02; int32_t  ne03;
    uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
    int32_t  ne0;  int32_t  ne1;  int32_t  ne2;  int32_t  ne3;
    uint64_t nb0;  uint64_t nb1;  uint64_t nb2;  uint64_t nb3;
    int32_t  n_past; int32_t  n_dims; int32_t  n_offs; int32_t  n_ctx_orig;
    float    freq_base; float freq_scale; float ext_factor; float attn_factor;
    float    beta_fast; float beta_slow;
    int32_t  sect_0; int32_t  sect_1; int32_t  sect_2; int32_t  sect_3;
    bool     src2; bool inplace;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

kernel void kernel_rope_neox_f32(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
            const int iw = i0 - args.n_offs;
            const int ic = iw/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);

            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const float * const src = (device float *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + (args.n_offs + ic)*args.nb00);
            device       float * dst_data  = (device float *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + (args.n_offs + ic)*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            if (args.inplace) {
                continue;
            }
            device const float * const src = (device float *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       float * dst_data  = (device float *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);
            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}
"#;

        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let kernel = library.get_function("kernel_rope_neox_f32", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        Ok(Self {
            device, queue, pipeline,
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

    fn rope(&self, src: &[f32], positions: &[i32], n_heads: usize, hidden_dim: usize, n_dims: usize, freq_base: f32, iterations: usize) -> f64 {
        let n_seqs = positions.len();
        let total_size = src.len();

        let input_buffer = self.get_buffer(total_size * 4, &self.input_buffer, &self.input_size);
        let output_buffer = self.get_buffer(total_size * 4, &self.output_buffer, &self.output_size);
        let pos_buffer = self.device.new_buffer((positions.len() * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), input_buffer.contents() as *mut f32, total_size);
            std::ptr::copy_nonoverlapping(positions.as_ptr(), pos_buffer.contents() as *mut i32, positions.len());
        }

        let args = LlamaRopeArgs {
            ne00: hidden_dim as i32, ne01: n_heads as i32, ne02: n_seqs as i32, ne03: 1,
            nb00: 4, nb01: (hidden_dim * 4) as u64, nb02: (n_heads * hidden_dim * 4) as u64, nb03: (n_seqs * n_heads * hidden_dim * 4) as u64,
            ne0: hidden_dim as i32, ne1: n_heads as i32, ne2: n_seqs as i32, ne3: 1,
            nb0: 4, nb1: (hidden_dim * 4) as u64, nb2: (n_heads * hidden_dim * 4) as u64, nb3: (n_seqs * n_heads * hidden_dim * 4) as u64,
            n_past: 0, n_dims: n_dims as i32, n_offs: 0, n_ctx_orig: 2048,
            freq_base, freq_scale: 1.0, ext_factor: 0.0, attn_factor: 1.0, beta_fast: 32.0, beta_slow: 1.0,
            sect_0: 0, sect_1: 0, sect_2: 0, sect_3: 0,
            src2: false, inplace: false,
        };

        let max_threads = self.pipeline.max_total_threads_per_threadgroup() as usize;
        let n_dims_half = n_dims / 2;
        let threads_per_tg = n_dims_half.min(max_threads).next_power_of_two();

        // Warmup
        for _ in 0..20 {
            let _ = self.dispatch(&input_buffer, &pos_buffer, &output_buffer, &args, n_heads, n_seqs, threads_per_tg);
        }

        // Measure
        let mut min_time = f64::MAX;
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.dispatch(&input_buffer, &pos_buffer, &output_buffer, &args, n_heads, n_seqs, threads_per_tg);
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed < min_time {
                min_time = elapsed;
            }
        }

        min_time
    }

    fn dispatch(&self, input: &Buffer, pos: &Buffer, output: &Buffer, args: &LlamaRopeArgs, n_heads: usize, n_seqs: usize, threads_per_tg: usize) -> Result<(), String> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_bytes(0, std::mem::size_of::<LlamaRopeArgs>() as u64, args as *const LlamaRopeArgs as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(pos), 0);
        encoder.set_buffer(3, None, 0);  // src2 = null
        encoder.set_buffer(4, Some(output), 0);

        let grid_size = MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: threads_per_tg as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

fn main() {
    let ours = OurRopeMetalContext::new().expect("Failed to create our context");
    let llama = LlamaCppRopeMetalContext::new().expect("Failed to create llama.cpp context");

    println!("=== Metal RoPE: Our Implementation vs llama.cpp ===\n");

    let configs: Vec<(usize, usize, usize)> = vec![
        (32, 128, 128),  // LLaMA-7B
        (32, 256, 128),  // LLaMA-13B style
        (64, 128, 128),  // LLaMA-70B
    ];

    let n_seqs = 128;
    let freq_base = 10000.0f32;
    let iterations = 5000;

    println!("Config (n_seqs={}):", n_seqs);
    println!("{:<15} {:>12} {:>12} {:>7}", "Config", "Ours (us)", "llama (us)", "Perf");
    println!("{}", "-".repeat(55));

    for (n_heads, hidden_dim, n_dims) in configs {
        let total_size = n_seqs * n_heads * hidden_dim;
        let src: Vec<f32> = (0..total_size).map(|i| ((i % 100) as f32 + 1.0) / 50.0).collect();
        let positions: Vec<i32> = (0..n_seqs as i32).collect();

        let t_ours = ours.rope(&src, &positions, n_heads, hidden_dim, n_dims, freq_base, iterations) * 1e6;
        let t_llama = llama.rope(&src, &positions, n_heads, hidden_dim, n_dims, freq_base, iterations) * 1e6;
        let ratio = t_llama / t_ours;

        println!("{:<15} {:>12.2} {:>12.2} {:>7.1}%",
            format!("{}h{}d", n_heads, hidden_dim), t_ours, t_llama, ratio * 100.0);
    }
}
