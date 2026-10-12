use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLSize,
};
use std::cell::RefCell;

pub struct RmsNormMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pipeline_vec4: ComputePipelineState,
    // Cached buffers
    input_buffer: RefCell<Option<Buffer>>,
    weight_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    weight_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

pub struct RopeMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    input_size: RefCell<usize>,
    output_size: RefCell<usize>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RopeArgs {
    ne00: i32,        // hidden_dim
    ne01: i32,        // n_heads * n_seqs
    ne02: i32,        // n_seqs
    nb00: u64,        // stride for dim
    nb01: u64,        // stride for head
    nb02: u64,        // stride for seq
    n_dims: i32,      // dimensions to rotate
    n_offs: i32,      // offset for rotation
    freq_base: f32,   // base frequency
    freq_scale: f32,  // frequency scaling
    ext_factor: f32,  // YaRN extension factor
    attn_factor: f32, // YaRN attention factor
    beta_fast: f32,   // YaRN fast beta
    beta_slow: f32,   // YaRN slow beta
    n_ctx_orig: i32,  // original context length
}

impl RmsNormMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("metal/rms_norm.metal");
        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile RMSNorm library: {}", e))?;

        let kernel = library
            .get_function("kernel_rms_norm_f32", None)
            .map_err(|e| format!("Failed to get RMSNorm kernel: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create RMSNorm pipeline: {}", e))?;

        // Vec4 kernel is optional - we'll fall back to scalar if not present
        let pipeline_vec4 =
            match library.get_function("kernel_rms_norm_f32_4", None) {
                Ok(kernel) => device
                    .new_compute_pipeline_state_with_function(&kernel)
                    .map_err(|e| {
                        format!("Failed to create RMSNorm vec4 pipeline: {}", e)
                    })?,
                Err(_) => pipeline.clone(), // Fallback to scalar pipeline
            };

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_vec4,
            input_buffer: RefCell::new(None),
            weight_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            weight_size: RefCell::new(0),
            output_size: RefCell::new(0),
        })
    }

    fn get_or_create_buffer(
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

    /// RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
    ///
    /// # Arguments
    /// * `x` - Input tensor [n_rows, hidden_dim]
    /// * `w` - Weight tensor [hidden_dim]
    /// * `hidden_dim` - Dimension of each row
    /// * `eps` - Small constant for numerical stability
    pub fn rms_norm_f32(
        &self,
        x: &[f32],
        w: &[f32],
        hidden_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let n_rows = x.len() / hidden_dim;
        let output_len = x.len();

        // Get or create buffers
        let input_buffer = self.get_or_create_buffer(
            x.len() * std::mem::size_of::<f32>(),
            &self.input_buffer,
            &self.input_size,
        );
        let weight_buffer = self.get_or_create_buffer(
            w.len() * std::mem::size_of::<f32>(),
            &self.weight_buffer,
            &self.weight_size,
        );
        let output_buffer = self.get_or_create_buffer(
            output_len * std::mem::size_of::<f32>(),
            &self.output_buffer,
            &self.output_size,
        );

        // Copy data to GPU
        unsafe {
            std::ptr::copy_nonoverlapping(
                x.as_ptr(),
                input_buffer.contents() as *mut f32,
                x.len(),
            );
            std::ptr::copy_nonoverlapping(
                w.as_ptr(),
                weight_buffer.contents() as *mut f32,
                w.len(),
            );
        }

        // Create command buffer
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Use vectorized version if hidden_dim is divisible by 4
        let use_vec4 = hidden_dim % 4 == 0;
        let pipeline =
            if use_vec4 { &self.pipeline_vec4 } else { &self.pipeline };

        encoder.set_compute_pipeline_state(pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        // Set constants
        let ne00 = hidden_dim as i32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &ne00 as *const i32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const std::ffi::c_void,
        );

        // Calculate threadgroup size - match llama.cpp's pattern
        // For vec4 kernel, use hidden_dim/4; for scalar, use hidden_dim
        let ne00_t = if use_vec4 { hidden_dim / 4 } else { hidden_dim };
        let max_threads_per_group =
            pipeline.max_total_threads_per_threadgroup() as usize;

        // Start at 32 (SIMD width) and double until >= ne00_t
        let mut threads_per_row = 32;
        while threads_per_row < ne00_t
            && threads_per_row < max_threads_per_group
        {
            threads_per_row *= 2;
        }
        // Cap at max threads and round up to nearest 32
        threads_per_row = threads_per_row.min(max_threads_per_group);
        threads_per_row = ((ne00_t + 31) / 32 * 32).min(threads_per_row);

        let threadgroup_size =
            MTLSize { width: threads_per_row as u64, height: 1, depth: 1 };

        let grid_size = MTLSize { width: n_rows as u64, height: 1, depth: 1 };

        // Allocate threadgroup memory for reduction
        // Need at least 32 floats (Metal SIMD width) for the two-stage reduction:
        // 1. First SIMD group initializes all 32 slots
        // 2. Each SIMD group writes to shmem_f32[sgitg]
        // 3. All threads read from shmem_f32[tiisg]
        let simd_groups_per_tg = (threads_per_row + 31) / 32;
        let shmem_size =
            std::cmp::max(32, simd_groups_per_tg) * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, shmem_size as u64); // index first, then size

        // Use threadgroup size to ensure proper reduction
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);

        encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results
        let mut output = vec![0.0f32; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                output_len,
            );
        }

        Ok(output)
    }
}

impl RopeMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("metal/rope.metal");
        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile RoPE library: {}", e))?;

        let kernel = library
            .get_function("kernel_rope_neox_f32", None)
            .map_err(|e| format!("Failed to get RoPE kernel: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create RoPE pipeline: {}", e))?;

        Ok(Self {
            device,
            queue,
            pipeline,
            input_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            output_size: RefCell::new(0),
        })
    }

    fn get_or_create_buffer(
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

    /// RoPE NeoX: apply rotary position embedding
    ///
    /// # Arguments
    /// * `src` - Input tensor [n_seqs, n_heads, hidden_dim]
    /// * `positions` - Position for each sequence
    /// * `n_dims` - Dimensions to rotate (usually = hidden_dim)
    /// * `freq_base` - Base frequency (usually 10000.0)
    pub fn rope_neox_f32(
        &self,
        src: &[f32],
        positions: &[i32],
        n_heads: usize,
        hidden_dim: usize,
        n_dims: usize,
        freq_base: f32,
    ) -> Result<Vec<f32>, String> {
        let n_seqs = positions.len();
        let total_size = src.len();

        // Get or create buffers
        let input_buffer = self.get_or_create_buffer(
            total_size * std::mem::size_of::<f32>(),
            &self.input_buffer,
            &self.input_size,
        );
        let output_buffer = self.get_or_create_buffer(
            total_size * std::mem::size_of::<f32>(),
            &self.output_buffer,
            &self.output_size,
        );

        // Position buffer (small, create fresh)
        let pos_buffer = self.device.new_buffer(
            positions.len() as u64 * std::mem::size_of::<i32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Copy data to GPU
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                input_buffer.contents() as *mut f32,
                total_size,
            );
            std::ptr::copy_nonoverlapping(
                positions.as_ptr(),
                pos_buffer.contents() as *mut i32,
                positions.len(),
            );
        }

        // Create command buffer
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set args first (buffer 0)
        let args = RopeArgs {
            ne00: hidden_dim as i32,
            ne01: n_heads as i32, // number of heads per row
            ne02: n_seqs as i32,
            nb00: std::mem::size_of::<f32>() as u64,
            nb01: hidden_dim as u64 * std::mem::size_of::<f32>() as u64,
            nb02: n_heads as u64
                * hidden_dim as u64
                * std::mem::size_of::<f32>() as u64,
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
        encoder.set_bytes(
            0,
            std::mem::size_of::<RopeArgs>() as u64,
            &args as *const RopeArgs as *const std::ffi::c_void,
        );

        // Set buffers (1, 2, 3)
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&pos_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        // Threadgroup: each threadgroup handles one (head, seq) pair
        let n_dims_half = n_dims / 2;
        let max_threads =
            self.pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_tg = n_dims_half.min(max_threads).next_power_of_two();

        let threadgroup_size =
            MTLSize { width: threads_per_tg as u64, height: 1, depth: 1 };

        let grid_size =
            MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results
        let mut output = vec![0.0f32; total_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                total_size,
            );
        }

        Ok(output)
    }
}

// ============== Softmax Metal Context ==============

pub struct SoftmaxMetalContext {
    device: Device,
    queue: CommandQueue,
    // Full-featured pipelines (support mask and ALiBi)
    pipeline: ComputePipelineState,
    pipeline_vec4: ComputePipelineState,
    // Fast path pipelines (no mask, no ALiBi)
    pipeline_fast: ComputePipelineState,
    pipeline_vec4_fast: ComputePipelineState,
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

impl SoftmaxMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("metal/softmax.metal");
        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile Softmax library: {}", e))?;

        let kernel = library
            .get_function("kernel_soft_max_f32", None)
            .map_err(|e| format!("Failed to get Softmax kernel: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create Softmax pipeline: {}", e))?;

        let kernel_vec4 =
            library.get_function("kernel_soft_max_f32_4", None).map_err(
                |e| format!("Failed to get Softmax vec4 kernel: {}", e),
            )?;
        let pipeline_vec4 = device
            .new_compute_pipeline_state_with_function(&kernel_vec4)
            .map_err(|e| {
                format!("Failed to create Softmax vec4 pipeline: {}", e)
            })?;

        // Fast path kernels
        let kernel_fast = library
            .get_function("kernel_soft_max_f32_fast", None)
            .map_err(|e| format!("Failed to get Softmax fast kernel: {}", e))?;
        let pipeline_fast = device
            .new_compute_pipeline_state_with_function(&kernel_fast)
            .map_err(|e| {
                format!("Failed to create Softmax fast pipeline: {}", e)
            })?;

        let kernel_vec4_fast =
            library.get_function("kernel_soft_max_f32_4_fast", None).map_err(
                |e| format!("Failed to get Softmax vec4 fast kernel: {}", e),
            )?;
        let pipeline_vec4_fast = device
            .new_compute_pipeline_state_with_function(&kernel_vec4_fast)
            .map_err(|e| {
                format!("Failed to create Softmax vec4 fast pipeline: {}", e)
            })?;

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_vec4,
            pipeline_fast,
            pipeline_vec4_fast,
            input_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            input_size: RefCell::new(0),
            output_size: RefCell::new(0),
        })
    }

    fn get_or_create_buffer(
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

    /// Softmax: apply softmax to each row
    ///
    /// # Arguments
    /// * `src` - Input tensor [batch, heads, seq_len]
    /// * `mask` - Optional mask tensor [batch, heads, seq_len] or [1, 1, seq_len]
    /// * `n_heads` - Number of attention heads
    /// * `seq_len` - Sequence length (row length)
    /// * `scale` - Temperature scaling factor
    /// * `max_bias` - ALiBi max bias (0 if not using ALiBi)
    pub fn softmax_f32(
        &self,
        src: &[f32],
        mask: Option<&[f32]>,
        n_heads: usize,
        seq_len: usize,
        scale: f32,
        max_bias: f32,
    ) -> Result<Vec<f32>, String> {
        let batch_size = src.len() / (n_heads * seq_len);
        let total_size = src.len();

        let input_buffer = self.get_or_create_buffer(
            total_size * std::mem::size_of::<f32>(),
            &self.input_buffer,
            &self.input_size,
        );
        let output_buffer = self.get_or_create_buffer(
            total_size * std::mem::size_of::<f32>(),
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

        // Mask buffer (optional)
        let mask_buffer = if let Some(m) = mask {
            let buf = self.device.new_buffer(
                m.len() as u64 * 4,
                metal::MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    m.as_ptr(),
                    buf.contents() as *mut f32,
                    m.len(),
                );
            }
            Some(buf)
        } else {
            None
        };

        // ALiBi parameters
        let (m0, m1, n_head_log2) = if max_bias > 0.0 {
            let n_head_log2 = (n_heads as f32).log2() as i32;
            let m0 = (-max_bias / n_head_log2 as f32).exp2();
            let m1 = (-max_bias / (2.0f32 * n_head_log2 as f32)).exp2();
            (m0, m1, n_head_log2)
        } else {
            (1.0f32, 1.0f32, 0)
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Use fast path if no mask, no ALiBi, and default scale
        let use_fast = mask.is_none() && max_bias == 0.0 && scale == 1.0;
        let use_vec4 = seq_len % 4 == 0;

        let pipeline = if use_fast {
            if use_vec4 {
                &self.pipeline_vec4_fast
            } else {
                &self.pipeline_fast
            }
        } else {
            if use_vec4 {
                &self.pipeline_vec4
            } else {
                &self.pipeline
            }
        };
        encoder.set_compute_pipeline_state(pipeline);

        if use_fast {
            // Fast path: just ne00, ne01
            let ne00 = seq_len as i32;
            let ne01 = n_heads as i32;
            encoder.set_bytes(
                0,
                4,
                &ne00 as *const i32 as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(
                3,
                4,
                &ne01 as *const i32 as *const std::ffi::c_void,
            );
        } else {
            // Full path: SoftmaxArgs struct
            let args = SoftmaxArgs {
                ne00: seq_len as i32,
                ne01: n_heads as i32,
                ne02: batch_size as i32,
                nb01: (seq_len * std::mem::size_of::<f32>()) as u64,
                nb02: (n_heads * seq_len * std::mem::size_of::<f32>()) as u64,
                nb03: 0,
                ne11: mask.as_ref().map(|m| m.len() / seq_len).unwrap_or(0)
                    as i32,
                ne12: n_heads as i32,
                ne13: batch_size as i32,
                nb11: (seq_len * std::mem::size_of::<f32>()) as u64,
                nb12: (n_heads * seq_len * std::mem::size_of::<f32>()) as u64,
                nb13: 0,
                nb1: (seq_len * std::mem::size_of::<f32>()) as u64,
                nb2: (n_heads * seq_len * std::mem::size_of::<f32>()) as u64,
                nb3: 0,
                scale,
                max_bias,
                m0,
                m1,
                n_head_log2,
            };

            encoder.set_bytes(
                0,
                std::mem::size_of::<SoftmaxArgs>() as u64,
                &args as *const SoftmaxArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(&input_buffer), 0);

            // Set mask buffer (or nullptr)
            if let Some(ref mb) = mask_buffer {
                encoder.set_buffer(2, Some(mb), 0);
            } else {
                // Use input_buffer as placeholder (shader checks src1 != src0)
                encoder.set_buffer(2, Some(&input_buffer), 0);
            }

            encoder.set_buffer(3, Some(&output_buffer), 0);
        }

        // Grid: one threadgroup per row
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_row = seq_len.min(max_threads).next_power_of_two();

        let grid_size = MTLSize {
            width: n_heads as u64,
            height: batch_size as u64,
            depth: 1,
        };
        let threadgroup_size =
            MTLSize { width: threads_per_row as u64, height: 1, depth: 1 };

        // Threadgroup memory for reduction
        let simd_groups = (threads_per_row + 31) / 32;
        let shmem_size = simd_groups * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut output = vec![0.0f32; total_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                total_size,
            );
        }

        Ok(output)
    }
}

// ============== Activation Metal Context ==============
// Hybrid dispatch strategy:
// - Small sizes (<32768): dispatch_threads (simpler, faster for small batches)
// - Large sizes (>=32768): dispatch_thread_groups with float4 (4 elements per thread)

const DISPATCH_THRESHOLD: usize = 32768;

pub struct ActivationMetalContext {
    device: Device,
    queue: CommandQueue,
    // dispatch_threads pipelines (for small sizes)
    silu_pipeline: ComputePipelineState,
    gelu_pipeline: ComputePipelineState,
    gelu_quick_pipeline: ComputePipelineState,
    // dispatch_thread_groups pipelines - scalar (for large sizes, n not divisible by 4)
    silu_pipeline_tg: ComputePipelineState,
    gelu_pipeline_tg: ComputePipelineState,
    gelu_quick_pipeline_tg: ComputePipelineState,
    // dispatch_thread_groups pipelines - float4 (for large sizes, n divisible by 4)
    silu_pipeline_4_tg: ComputePipelineState,
    gelu_pipeline_4_tg: ComputePipelineState,
    gelu_quick_pipeline_4_tg: ComputePipelineState,
    input_buffer: RefCell<Option<Buffer>>,
    output_buffer: RefCell<Option<Buffer>>,
    buffer_size: RefCell<usize>,
}

impl ActivationMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("metal/activation.metal");
        let library =
            device.new_library_with_source(source, &compile_options).map_err(
                |e| format!("Failed to compile activation library: {}", e),
            )?;

        // dispatch_threads kernels
        let silu_pipeline =
            create_pipeline(&device, &library, "kernel_silu_f32")?;
        let gelu_pipeline =
            create_pipeline(&device, &library, "kernel_gelu_f32")?;
        let gelu_quick_pipeline =
            create_pipeline(&device, &library, "kernel_gelu_quick_f32")?;

        // dispatch_thread_groups kernels - scalar
        let silu_pipeline_tg =
            create_pipeline(&device, &library, "kernel_silu_f32_tg")?;
        let gelu_pipeline_tg =
            create_pipeline(&device, &library, "kernel_gelu_f32_tg")?;
        let gelu_quick_pipeline_tg =
            create_pipeline(&device, &library, "kernel_gelu_quick_f32_tg")?;

        // dispatch_thread_groups kernels - float4
        let silu_pipeline_4_tg =
            create_pipeline(&device, &library, "kernel_silu_f32_4_tg")?;
        let gelu_pipeline_4_tg =
            create_pipeline(&device, &library, "kernel_gelu_f32_4_tg")?;
        let gelu_quick_pipeline_4_tg =
            create_pipeline(&device, &library, "kernel_gelu_quick_f32_4_tg")?;

        Ok(Self {
            device,
            queue,
            silu_pipeline,
            gelu_pipeline,
            gelu_quick_pipeline,
            silu_pipeline_tg,
            gelu_pipeline_tg,
            gelu_quick_pipeline_tg,
            silu_pipeline_4_tg,
            gelu_pipeline_4_tg,
            gelu_quick_pipeline_4_tg,
            input_buffer: RefCell::new(None),
            output_buffer: RefCell::new(None),
            buffer_size: RefCell::new(0),
        })
    }

    fn get_or_create_buffer(&self, size: usize) -> (Buffer, Buffer) {
        let mut buffer_size = self.buffer_size.borrow_mut();
        let mut input_buffer = self.input_buffer.borrow_mut();
        let mut output_buffer = self.output_buffer.borrow_mut();

        if *buffer_size != size {
            *input_buffer = Some(self.device.new_buffer(
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            *output_buffer = Some(self.device.new_buffer(
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            *buffer_size = size;
        }

        (
            input_buffer.as_ref().unwrap().clone(),
            output_buffer.as_ref().unwrap().clone(),
        )
    }

    pub fn silu(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let n = input.len();
        if n < DISPATCH_THRESHOLD {
            self.run_activation_dispatch_threads(&self.silu_pipeline, input)
        } else if n % 4 == 0 {
            self.run_activation_dispatch_thread_groups_4(
                &self.silu_pipeline_4_tg,
                input,
            )
        } else {
            self.run_activation_dispatch_thread_groups(
                &self.silu_pipeline_tg,
                input,
            )
        }
    }

    pub fn gelu(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let n = input.len();
        if n < DISPATCH_THRESHOLD {
            self.run_activation_dispatch_threads(&self.gelu_pipeline, input)
        } else if n % 4 == 0 {
            self.run_activation_dispatch_thread_groups_4(
                &self.gelu_pipeline_4_tg,
                input,
            )
        } else {
            self.run_activation_dispatch_thread_groups(
                &self.gelu_pipeline_tg,
                input,
            )
        }
    }

    pub fn gelu_quick(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let n = input.len();
        if n < DISPATCH_THRESHOLD {
            self.run_activation_dispatch_threads(
                &self.gelu_quick_pipeline,
                input,
            )
        } else if n % 4 == 0 {
            self.run_activation_dispatch_thread_groups_4(
                &self.gelu_quick_pipeline_4_tg,
                input,
            )
        } else {
            self.run_activation_dispatch_thread_groups(
                &self.gelu_quick_pipeline_tg,
                input,
            )
        }
    }

    fn run_activation_dispatch_threads(
        &self,
        pipeline: &ComputePipelineState,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let n = input.len();
        let size = n * std::mem::size_of::<f32>();
        let (input_buffer, output_buffer) = self.get_or_create_buffer(size);

        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_buffer.contents() as *mut f32,
                n,
            );
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let threadgroup_size = MTLSize { width: 256, height: 1, depth: 1 };
        let grid_size = MTLSize { width: n as u64, height: 1, depth: 1 };
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut output = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                n,
            );
        }

        Ok(output)
    }

    fn run_activation_dispatch_thread_groups(
        &self,
        pipeline: &ComputePipelineState,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let n = input.len();
        let ne00 = n as i32;
        let size = n * std::mem::size_of::<f32>();
        let (input_buffer, output_buffer) = self.get_or_create_buffer(size);

        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_buffer.contents() as *mut f32,
                n,
            );
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        let threads_per_group = 256;
        let num_groups = (n + threads_per_group - 1) / threads_per_group;
        let threadgroup_size =
            MTLSize { width: threads_per_group as u64, height: 1, depth: 1 };
        let grid_size =
            MTLSize { width: num_groups as u64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut output = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                n,
            );
        }

        Ok(output)
    }

    fn run_activation_dispatch_thread_groups_4(
        &self,
        pipeline: &ComputePipelineState,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let n = input.len();
        let ne00 = n as i32;
        let size = n * std::mem::size_of::<f32>();
        let (input_buffer, output_buffer) = self.get_or_create_buffer(size);

        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_buffer.contents() as *mut f32,
                n,
            );
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        // float4 kernel: each thread processes 4 elements
        let n4 = n / 4;
        let threads_per_group = 256;
        let num_groups = (n4 + threads_per_group - 1) / threads_per_group;
        let threadgroup_size =
            MTLSize { width: threads_per_group as u64, height: 1, depth: 1 };
        let grid_size =
            MTLSize { width: num_groups as u64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut output = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                n,
            );
        }

        Ok(output)
    }
}

fn create_pipeline(
    device: &Device,
    library: &metal::LibraryRef,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let kernel = library
        .get_function(name, None)
        .map_err(|e| format!("Failed to get {}: {}", name, e))?;
    device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Failed to create pipeline for {}: {}", name, e))
}

fn create_pipeline_with_constants(
    device: &Device,
    library: &metal::LibraryRef,
    name: &str,
    constant_values: &metal::FunctionConstantValues,
) -> Result<ComputePipelineState, String> {
    let kernel = library
        .get_function(name, Some(constant_values.clone()))
        .map_err(|e| format!("Failed to get {} with constants: {}", name, e))?;
    device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Failed to create pipeline for {}: {}", name, e))
}

// ============================================================================
// Flash Attention Metal Context
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct FlashAttnArgs {
    ne01: u32,     // Q rows
    ne02: u32,     // Q heads
    ne03: u32,     // Q batches
    ne11: u32,     // K rows (seq_len)
    ne_12_2: u32,  // KV heads
    ne_12_3: u32,  // KV batches
    nb01: u32,     // Q stride (bytes)
    nb02: u32,     // Q stride (bytes)
    nb03: u32,     // Q stride (bytes)
    nb11: u32,     // K stride (bytes)
    nb12: u32,     // K stride (bytes)
    nb13: u32,     // K stride (bytes)
    nb21: u32,     // V stride (bytes)
    nb22: u32,     // V stride (bytes)
    nb23: u32,     // V stride (bytes)
    nb31: u32,     // mask stride
    nb32: u32,     // mask stride
    nb33: u32,     // mask stride
    ne32: u32,     // mask rows
    ne33: u32,     // mask batches
    scale: f32,    // 1/sqrt(head_dim)
    max_bias: f32, // ALiBi
    logit_softcap: f32,
    n_head_log2: u32,
    m0: f32,
    m1: f32,
    nsg: u32, // simdgroups per threadgroup
    nwg: u32, // workgroups (for parallelization across seq_len)
}

/// KV cache format type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KvFormat {
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
}

pub struct FlashAttnMetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,
    // Pipelines for each format (head_dim=128)
    pipeline_f16: ComputePipelineState,
    pipeline_q4_0: ComputePipelineState,
    pipeline_q4_1: ComputePipelineState,
    pipeline_q5_0: ComputePipelineState,
    pipeline_q5_1: ComputePipelineState,
    pipeline_q8_0: ComputePipelineState,
    // Cached buffers
    q_buffer: RefCell<Option<Buffer>>,
    k_buffer: RefCell<Option<Buffer>>,
    v_buffer: RefCell<Option<Buffer>>,
    mask_buffer: RefCell<Option<Buffer>>,
    dst_buffer: RefCell<Option<Buffer>>,
    shmem_size: RefCell<usize>,
}

impl FlashAttnMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        // Combine dequantize.metal and flash_attn.metal
        let dequantize_source = include_str!("metal/dequantize.metal");
        let flash_attn_source = include_str!("metal/flash_attn.metal");
        // Remove the include directive and concatenate
        let source = format!(
            "{}\n{}",
            dequantize_source,
            flash_attn_source.replace("#include \"dequantize.metal\"", "")
        );
        let source = source.as_str();

        let library =
            device.new_library_with_source(source, &compile_options).map_err(
                |e| format!("Failed to compile Flash Attention library: {}", e),
            )?;

        // Create function constants for vec kernel (decode case)
        // Using llama.cpp's constant indices (FC_FLASH_ATTN_EXT_VEC = 1000)
        // FC_FLASH_ATTN_EXT_VEC + 0 = has_mask
        // FC_FLASH_ATTN_EXT_VEC + 1 = has_sinks
        // FC_FLASH_ATTN_EXT_VEC + 2 = has_bias
        // FC_FLASH_ATTN_EXT_VEC + 3 = has_scap
        // FC_FLASH_ATTN_EXT_VEC + 4 = has_kvpad
        // FC_FLASH_ATTN_EXT_VEC + 20 = ns10
        // FC_FLASH_ATTN_EXT_VEC + 21 = ns20
        // FC_FLASH_ATTN_EXT_VEC + 22 = nsg
        // FC_FLASH_ATTN_EXT_VEC + 23 = nwg
        let constant_values = metal::FunctionConstantValues::new();
        // FC_has_mask = false (index 0)
        let has_mask = false;
        constant_values.set_constant_value_at_index(
            &has_mask as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            0,
        );
        // FC_has_sinks = false (index 1)
        let has_sinks = false;
        constant_values.set_constant_value_at_index(
            &has_sinks as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            1,
        );
        // FC_has_bias = false (index 2)
        let has_bias = false;
        constant_values.set_constant_value_at_index(
            &has_bias as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            2,
        );
        // FC_has_scap = false (index 3)
        let has_scap = false;
        constant_values.set_constant_value_at_index(
            &has_scap as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            3,
        );
        // FC_has_kvpad = false (index 4)
        let has_kvpad = false;
        constant_values.set_constant_value_at_index(
            &has_kvpad as *const bool as *const std::ffi::c_void,
            metal::MTLDataType::Bool,
            4,
        );
        // FC_nsg = 1 (index 22, number of SIMD groups per threadgroup)
        let nsg: i32 = 1;
        constant_values.set_constant_value_at_index(
            &nsg as *const i32 as *const std::ffi::c_void,
            metal::MTLDataType::Int,
            22,
        );
        // FC_nwg = 1 (index 23, single workgroup for direct output)
        let nwg: i32 = 1;
        constant_values.set_constant_value_at_index(
            &nwg as *const i32 as *const std::ffi::c_void,
            metal::MTLDataType::Int,
            23,
        );
        // FC_ns10 = nb11/nb10 = number of elements per row (index 20)
        // For F16: nb10_attn = sizeof(half) = 2, nb11_attn = head_dim * sizeof(half) = 256
        // ns10 = nb11_attn / nb10_attn = 256 / 2 = 128 (number of half elements per row)
        // In the kernel: NS10/4 = 32 half4 elements per row
        let ns10: i32 = 128; // 128 half elements for head_dim=128
        constant_values.set_constant_value_at_index(
            &ns10 as *const i32 as *const std::ffi::c_void,
            metal::MTLDataType::Int,
            20,
        );
        // FC_ns20 = nb21/nb20 (index 21)
        let ns20: i32 = 128; // Same as ns10 for F16
        constant_values.set_constant_value_at_index(
            &ns20 as *const i32 as *const std::ffi::c_void,
            metal::MTLDataType::Int,
            21,
        );

        // Create pipelines for each format (head_dim=128)
        let pipeline_f16 = create_pipeline_with_constants(
            &device,
            &library,
            "kernel_flash_attn_ext_vec_f16_dk128_dv128",
            &constant_values,
        )?;
        let pipeline_q4_0 = create_pipeline_with_constants(
            &device,
            &library,
            "kernel_flash_attn_ext_vec_q4_0_dk128_dv128",
            &constant_values,
        )?;
        let pipeline_q4_1 = create_pipeline_with_constants(
            &device,
            &library,
            "kernel_flash_attn_ext_vec_q4_1_dk128_dv128",
            &constant_values,
        )?;
        let pipeline_q5_0 = create_pipeline_with_constants(
            &device,
            &library,
            "kernel_flash_attn_ext_vec_q5_0_dk128_dv128",
            &constant_values,
        )?;
        let pipeline_q5_1 = create_pipeline_with_constants(
            &device,
            &library,
            "kernel_flash_attn_ext_vec_q5_1_dk128_dv128",
            &constant_values,
        )?;
        let pipeline_q8_0 = create_pipeline_with_constants(
            &device,
            &library,
            "kernel_flash_attn_ext_vec_q8_0_dk128_dv128",
            &constant_values,
        )?;

        Ok(Self {
            device,
            queue,
            library,
            pipeline_f16,
            pipeline_q4_0,
            pipeline_q4_1,
            pipeline_q5_0,
            pipeline_q5_1,
            pipeline_q8_0,
            q_buffer: RefCell::new(None),
            k_buffer: RefCell::new(None),
            v_buffer: RefCell::new(None),
            mask_buffer: RefCell::new(None),
            dst_buffer: RefCell::new(None),
            shmem_size: RefCell::new(0),
        })
    }

    fn get_pipeline(&self, format: KvFormat) -> &ComputePipelineState {
        match format {
            KvFormat::F16 => &self.pipeline_f16,
            KvFormat::Q4_0 => &self.pipeline_q4_0,
            KvFormat::Q4_1 => &self.pipeline_q4_1,
            KvFormat::Q5_0 => &self.pipeline_q5_0,
            KvFormat::Q5_1 => &self.pipeline_q5_1,
            KvFormat::Q8_0 => &self.pipeline_q8_0,
        }
    }

    fn get_or_create_buffer(
        &self,
        size: usize,
        buffer_ref: &RefCell<Option<Buffer>>,
    ) -> Buffer {
        let mut buffer_cell = buffer_ref.borrow_mut();
        if buffer_cell.is_none()
            || buffer_cell.as_ref().unwrap().length() as usize != size
        {
            *buffer_cell = Some(self.device.new_buffer(
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
        }
        buffer_cell.as_ref().unwrap().clone()
    }

    /// Flash Attention for decode (batch < 20)
    ///
    /// # Arguments
    /// * `q` - Query tensor [n_heads, head_dim] (single query)
    /// * `k` - Key tensor [seq_len, n_kv_heads, head_dim] in quantized format
    /// * `v` - Value tensor [seq_len, n_kv_heads, head_dim] in quantized format
    /// * `mask` - Optional attention mask [seq_len]
    /// * `n_heads` - Number of query heads
    /// * `n_kv_heads` - Number of KV heads (for GQA)
    /// * `head_dim` - Head dimension (must be 128 for now)
    /// * `seq_len` - Sequence length
    /// * `format` - KV cache format
    ///
    /// # Returns
    /// Output tensor [n_heads, head_dim]
    pub fn flash_attn_decode(
        &self,
        q: &[f32],
        k: &[u8],
        v: &[u8],
        mask: Option<&[f32]>,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        format: KvFormat,
    ) -> Result<Vec<f32>, String> {
        if head_dim != 128 {
            return Err(format!(
                "Flash Attention only supports head_dim=128, got {}",
                head_dim
            ));
        }

        let pipeline = self.get_pipeline(format);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Calculate buffer sizes
        let q_size = q.len() * std::mem::size_of::<f32>();
        let k_size = k.len();
        let v_size = v.len();
        let mask_size =
            mask.map(|m| m.len() * std::mem::size_of::<f32>()).unwrap_or(0);
        let dst_size =
            n_heads as usize * head_dim as usize * std::mem::size_of::<f32>();

        // Get or create buffers
        let q_buffer = self.get_or_create_buffer(q_size, &self.q_buffer);
        let k_buffer = self.get_or_create_buffer(k_size, &self.k_buffer);
        let v_buffer = self.get_or_create_buffer(v_size, &self.v_buffer);
        let mask_buffer = if mask.is_some() {
            Some(self.get_or_create_buffer(mask_size, &self.mask_buffer))
        } else {
            None
        };
        let dst_buffer = self.get_or_create_buffer(dst_size, &self.dst_buffer);

        // Copy data to GPU
        unsafe {
            std::ptr::copy_nonoverlapping(
                q.as_ptr(),
                q_buffer.contents() as *mut f32,
                q.len(),
            );
            std::ptr::copy_nonoverlapping(
                k.as_ptr(),
                k_buffer.contents() as *mut u8,
                k.len(),
            );
            std::ptr::copy_nonoverlapping(
                v.as_ptr(),
                v_buffer.contents() as *mut u8,
                v.len(),
            );
            if let (Some(m), Some(mb)) = (mask, mask_buffer.as_ref()) {
                std::ptr::copy_nonoverlapping(
                    m.as_ptr(),
                    mb.contents() as *mut f32,
                    m.len(),
                );
            }
        }

        // Setup arguments (matching llama.cpp's layout)
        // Q layout for decode: [n_heads, head_dim] with ne01=1, ne02=n_heads
        // The test passes q as a flat [n_heads * head_dim] array
        // Strides: nb01 = head_dim * 4, nb02 = head_dim * 4, nb03 = n_heads * head_dim * 4
        let ne01 = 1; // Single query for decode (not n_heads!)

        // Calculate parallelization parameters (matching llama.cpp)
        let ncpsg: u32 = 32; // cache items per simdgroup (C parameter in kernel)
        let mut nwg: u32 = 32; // workgroups
        let mut nsg: u32 = 1; // simdgroups per workgroup

        // Adjust nsg based on seq_len (matching llama.cpp's logic)
        // Double nsg while 2*nwg*nsg*ncpsg < seq_len and nsg < 4
        while 2 * nwg * nsg * ncpsg < seq_len && nsg < 4 {
            nsg *= 2;
        }

        let args = FlashAttnArgs {
            ne01: ne01,                   // Q rows (1 for decode)
            ne02: n_heads,                // Q heads
            ne03: 1,                      // Q batches (single batch for decode)
            ne11: seq_len,                // K rows (seq_len)
            ne_12_2: n_kv_heads,          // KV heads
            ne_12_3: 1,                   // KV batches
            nb01: head_dim * 4,           // Q stride for one head (bytes)
            nb02: head_dim * 4, // Q stride for one query (same as nb01 for ne01=1)
            nb03: n_heads * head_dim * 4, // Q stride for one batch
            nb11: head_dim * 2, // K stride (F16: head_dim * sizeof(half))
            nb12: head_dim * 2 * seq_len,
            nb13: head_dim * 2 * seq_len * n_kv_heads,
            nb21: head_dim * 2, // V stride
            nb22: head_dim * 2 * seq_len,
            nb23: head_dim * 2 * seq_len * n_kv_heads,
            nb31: seq_len * 2, // mask stride
            nb32: seq_len * 2,
            nb33: seq_len * 2,
            ne32: 1, // mask rows
            ne33: 1, // mask batches
            scale,
            max_bias: 0.0,
            logit_softcap: 0.0,
            n_head_log2: 0,
            m0: 0.0,
            m1: 0.0,
            nsg,
            nwg,
        };

        // Create command buffer
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        // Set args as buffer 0 (using set_bytes)
        encoder.set_bytes(
            0,
            std::mem::size_of::<FlashAttnArgs>() as u64,
            &args as *const FlashAttnArgs as *const std::ffi::c_void,
        );

        // Set buffers (indices match kernel signature)
        // buffer 1: q
        encoder.set_buffer(1, Some(&q_buffer), 0);
        // buffer 2: k
        encoder.set_buffer(2, Some(&k_buffer), 0);
        // buffer 3: v
        encoder.set_buffer(3, Some(&v_buffer), 0);
        // buffer 4: mask (optional)
        if let Some(ref mb) = mask_buffer {
            encoder.set_buffer(4, Some(mb), 0);
        }
        // buffer 5: sinks (not used, set to null)
        // buffer 6: pad (not used, set to null)
        // buffer 7: dst
        encoder.set_buffer(7, Some(&dst_buffer), 0);

        // Calculate grid and threadgroup sizes (matching llama.cpp's parallelization)
        // Grid: [(ne01 + Q - 1) / Q, ne02, ne03 * nwg]
        // For decode: ne01=1, Q=1, ne02=n_heads, ne03=1
        // Grid becomes: [1, n_heads, nwg]
        let nqptg = 1; // Q = 1 query per threadgroup
        let grid_size = MTLSize {
            width: ((ne01 + nqptg - 1) / nqptg) as u64, // 1 for decode
            height: n_heads as u64,                     // n_heads
            depth: args.nwg as u64,                     // nwg workgroups
        };

        // Threadgroup size: 32 (SIMD width) * nsg
        let threadgroup_size =
            MTLSize { width: (32 * args.nsg) as u64, height: 1, depth: 1 };

        // Allocate threadgroup memory
        // Shared memory layout (matches llama.cpp):
        // For C=32, NE=1:
        // - sq4:  Q * PK * sizeof(half)  = 1 * 128 * 2 = 256 bytes
        // - ss:   NSG * SH * sizeof(float) where SH = 4*Q*C = 4*1*32 = 128 floats = 512 bytes
        // - sm:   Q * C * sizeof(half) = 1 * 32 * 2 = 64 bytes
        // - so4:  NSG * Q * PV * sizeof(float4) = 1 * 1 * 128 * 16 = 2048 bytes
        // Total: 256 + 512 + 64 + 2048 = 2880 bytes
        let shmem_size = 2880;
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back
        let output_len = n_heads as usize * head_dim as usize;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                dst_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                output_len,
            );
        }

        Ok(output)
    }
}
