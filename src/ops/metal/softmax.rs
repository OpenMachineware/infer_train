use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLSize};
use std::cell::RefCell;

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

        let source = include_str!("ops/softmax.metal");
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
