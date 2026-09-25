use metal::{Device, CommandQueue, Library, ComputePipelineState, MTLSize, Buffer};
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
    ne00: i32,       // hidden_dim
    ne01: i32,       // n_heads * n_seqs
    ne02: i32,       // n_seqs
    nb00: u64,       // stride for dim
    nb01: u64,       // stride for head
    nb02: u64,       // stride for seq
    n_dims: i32,     // dimensions to rotate
    n_offs: i32,     // offset for rotation
    freq_base: f32,  // base frequency
    freq_scale: f32, // frequency scaling
    ext_factor: f32, // YaRN extension factor
    attn_factor: f32,// YaRN attention factor
    beta_fast: f32,  // YaRN fast beta
    beta_slow: f32,  // YaRN slow beta
    n_ctx_orig: i32, // original context length
}

impl RmsNormMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("../../../shaders/rms_norm.metal");
        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile RMSNorm library: {}", e))?;

        let kernel = library.get_function("kernel_rms_norm_f32", None)
            .map_err(|e| format!("Failed to get RMSNorm kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create RMSNorm pipeline: {}", e))?;

        // Vec4 kernel is optional - we'll fall back to scalar if not present
        let pipeline_vec4 = match library.get_function("kernel_rms_norm_f32_4", None) {
            Ok(kernel) => device.new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| format!("Failed to create RMSNorm vec4 pipeline: {}", e))?,
            Err(_) => pipeline.clone(),  // Fallback to scalar pipeline
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
            *buffer_cell = Some(self.device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared));
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
        let pipeline = if use_vec4 {
            &self.pipeline_vec4
        } else {
            &self.pipeline
        };

        encoder.set_compute_pipeline_state(pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        // Set constants
        let ne00 = hidden_dim as i32;
        encoder.set_bytes(3, std::mem::size_of::<i32>() as u64, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_bytes(4, std::mem::size_of::<f32>() as u64, &eps as *const f32 as *const std::ffi::c_void);

        // Calculate threadgroup size - match llama.cpp's pattern
        // For vec4 kernel, use hidden_dim/4; for scalar, use hidden_dim
        let ne00_t = if use_vec4 { hidden_dim / 4 } else { hidden_dim };
        let max_threads_per_group = pipeline.max_total_threads_per_threadgroup() as usize;

        // Start at 32 (SIMD width) and double until >= ne00_t
        let mut threads_per_row = 32;
        while threads_per_row < ne00_t && threads_per_row < max_threads_per_group {
            threads_per_row *= 2;
        }
        // Cap at max threads and round up to nearest 32
        threads_per_row = threads_per_row.min(max_threads_per_group);
        threads_per_row = ((ne00_t + 31) / 32 * 32).min(threads_per_row);

        let threadgroup_size = MTLSize {
            width: threads_per_row as u64,
            height: 1,
            depth: 1,
        };

        let grid_size = MTLSize {
            width: n_rows as u64,
            height: 1,
            depth: 1,
        };

        // Allocate threadgroup memory for reduction
        // Need at least 32 floats (Metal SIMD width) for the two-stage reduction:
        // 1. First SIMD group initializes all 32 slots
        // 2. Each SIMD group writes to shmem_f32[sgitg]
        // 3. All threads read from shmem_f32[tiisg]
        let simd_groups_per_tg = (threads_per_row + 31) / 32;
        let shmem_size = std::cmp::max(32, simd_groups_per_tg) * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);  // index first, then size

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
        let device = Device::system_default()
            .ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("../../../shaders/rope.metal");
        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile RoPE library: {}", e))?;

        let kernel = library.get_function("kernel_rope_neox_f32", None)
            .map_err(|e| format!("Failed to get RoPE kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
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
            *buffer_cell = Some(self.device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared));
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
            metal::MTLResourceOptions::StorageModeShared
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
            ne01: n_heads as i32,  // number of heads per row
            ne02: n_seqs as i32,
            nb00: std::mem::size_of::<f32>() as u64,
            nb01: hidden_dim as u64 * std::mem::size_of::<f32>() as u64,
            nb02: n_heads as u64 * hidden_dim as u64 * std::mem::size_of::<f32>() as u64,
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
        encoder.set_bytes(0, std::mem::size_of::<RopeArgs>() as u64, &args as *const RopeArgs as *const std::ffi::c_void);

        // Set buffers (1, 2, 3)
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&pos_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        // Threadgroup: each threadgroup handles one (head, seq) pair
        let n_dims_half = n_dims / 2;
        let max_threads = self.pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_tg = n_dims_half.min(max_threads).next_power_of_two();

        let threadgroup_size = MTLSize {
            width: threads_per_tg as u64,
            height: 1,
            depth: 1,
        };

        let grid_size = MTLSize {
            width: n_heads as u64,
            height: n_seqs as u64,
            depth: 1,
        };

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

        let source = include_str!("../../../shaders/softmax.metal");
        let library = device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile Softmax library: {}", e))?;

        let kernel = library.get_function("kernel_soft_max_f32", None)
            .map_err(|e| format!("Failed to get Softmax kernel: {}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create Softmax pipeline: {}", e))?;

        let kernel_vec4 = library.get_function("kernel_soft_max_f32_4", None)
            .map_err(|e| format!("Failed to get Softmax vec4 kernel: {}", e))?;
        let pipeline_vec4 = device.new_compute_pipeline_state_with_function(&kernel_vec4)
            .map_err(|e| format!("Failed to create Softmax vec4 pipeline: {}", e))?;

        // Fast path kernels
        let kernel_fast = library.get_function("kernel_soft_max_f32_fast", None)
            .map_err(|e| format!("Failed to get Softmax fast kernel: {}", e))?;
        let pipeline_fast = device.new_compute_pipeline_state_with_function(&kernel_fast)
            .map_err(|e| format!("Failed to create Softmax fast pipeline: {}", e))?;

        let kernel_vec4_fast = library.get_function("kernel_soft_max_f32_4_fast", None)
            .map_err(|e| format!("Failed to get Softmax vec4 fast kernel: {}", e))?;
        let pipeline_vec4_fast = device.new_compute_pipeline_state_with_function(&kernel_vec4_fast)
            .map_err(|e| format!("Failed to create Softmax vec4 fast pipeline: {}", e))?;

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
            *buffer_cell = Some(self.device.new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared));
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
            let buf = self.device.new_buffer(m.len() as u64 * 4, metal::MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(m.as_ptr(), buf.contents() as *mut f32, m.len());
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
            if use_vec4 { &self.pipeline_vec4_fast } else { &self.pipeline_fast }
        } else {
            if use_vec4 { &self.pipeline_vec4 } else { &self.pipeline }
        };
        encoder.set_compute_pipeline_state(pipeline);

        if use_fast {
            // Fast path: just ne00, ne01
            let ne00 = seq_len as i32;
            let ne01 = n_heads as i32;
            encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(3, 4, &ne01 as *const i32 as *const std::ffi::c_void);
        } else {
            // Full path: SoftmaxArgs struct
            let args = SoftmaxArgs {
                ne00: seq_len as i32,
                ne01: n_heads as i32,
                ne02: batch_size as i32,
                nb01: (seq_len * std::mem::size_of::<f32>()) as u64,
                nb02: (n_heads * seq_len * std::mem::size_of::<f32>()) as u64,
                nb03: 0,
                ne11: mask.as_ref().map(|m| m.len() / seq_len).unwrap_or(0) as i32,
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

            encoder.set_bytes(0, std::mem::size_of::<SoftmaxArgs>() as u64, &args as *const SoftmaxArgs as *const std::ffi::c_void);
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

        let grid_size = MTLSize { width: n_heads as u64, height: batch_size as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: threads_per_row as u64, height: 1, depth: 1 };

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
