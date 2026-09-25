use metal::{Device, CommandQueue, Library, ComputePipelineState, MTLSize, Buffer};
use std::cell::RefCell;

pub struct RmsNormMetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,
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
            library,
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
