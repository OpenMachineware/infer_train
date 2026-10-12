// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLSize};
use std::cell::RefCell;

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

impl RopeMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("ops/rope.metal");
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
