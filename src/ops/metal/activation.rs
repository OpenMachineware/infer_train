// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLSize};
use std::cell::RefCell;

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

        let source = include_str!("ops/activation.metal");
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
