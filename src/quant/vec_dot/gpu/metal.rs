use metal::{Device, CommandQueue, Library, ComputePipelineState, MTLSize, Buffer};
use crate::quant::types::{BlockIQ4NL, BlockQ8_0, BlockQ4_0, BlockQ4K, BlockQ5K, BlockQ2K, BlockQ6K, BlockQ3K};
use std::cell::RefCell;

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    vec_dot_library: Library,
    mv_library: Library,
    mv_simd_library: Library,
    mv_q4_0_library: Library,
    mv_q4_k_library: Library,
    mv_q5_k_library: Library,
    mv_q2_k_library: Library,
    mv_q6_k_library: Library,
    mv_q3_k_library: Library,
    mv_iq_tq_library: Library,
    // Cached pipelines
    mv_q4_0_pipeline: ComputePipelineState,
    mv_q4_k_pipeline: ComputePipelineState,
    mv_q5_k_pipeline: ComputePipelineState,
    mv_q2_k_pipeline: ComputePipelineState,
    mv_q6_k_pipeline: ComputePipelineState,
    mv_q3_k_pipeline: ComputePipelineState,
    mv_iq4_nl_pipeline: ComputePipelineState,
    // Cached buffers for MV operations
    // Once created with data, reuse without copy
    mv_weights_buffer: RefCell<Option<Buffer>>,
    mv_input_buffer: RefCell<Option<Buffer>>,
    mv_output_buffer: RefCell<Option<Buffer>>,
    // Track buffer sizes to know if we need to recreate
    mv_weights_size: RefCell<usize>,
    mv_input_size: RefCell<usize>,
    mv_output_size: RefCell<usize>,
}

impl MetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();

        let vec_dot_source = include_str!("../../../../shaders/vec_dot.metal");
        let vec_dot_library = device
            .new_library_with_source(vec_dot_source, &compile_options)
            .map_err(|e| format!("Failed to compile vec_dot library: {}", e))?;

        let mv_source = include_str!("../../../../shaders/mv.metal");
        let mv_library = device
            .new_library_with_source(mv_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv library: {}", e))?;

        let mv_simd_source = include_str!("../../../../shaders/mv_simd.metal");
        let mv_simd_library = device
            .new_library_with_source(mv_simd_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_simd library: {}", e))?;

        let mv_q4_0_source = include_str!("../../../../shaders/mv_q4_0.metal");
        let mv_q4_0_library = device
            .new_library_with_source(mv_q4_0_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q4_0 library: {}", e))?;

        let mv_q4_k_source = include_str!("../../../../shaders/mv_q4_k.metal");
        let mv_q4_k_library = device
            .new_library_with_source(mv_q4_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q4_k library: {}", e))?;

        let mv_q5_k_source = include_str!("../../../../shaders/mv_q5_k.metal");
        let mv_q5_k_library = device
            .new_library_with_source(mv_q5_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q5_k library: {}", e))?;

        let mv_q2_k_source = include_str!("../../../../shaders/mv_q2_k.metal");
        let mv_q2_k_library = device
            .new_library_with_source(mv_q2_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q2_k library: {}", e))?;

        let mv_q6_k_source = include_str!("../../../../shaders/mv_q6_k.metal");
        let mv_q6_k_library = device
            .new_library_with_source(mv_q6_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q6_k library: {}", e))?;

        let mv_q3_k_source = include_str!("../../../../shaders/mv_q3_k.metal");
        let mv_q3_k_library = device
            .new_library_with_source(mv_q3_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q3_k library: {}", e))?;

        let mv_iq_tq_source = include_str!("../../../../shaders/mv_iq_tq.metal");
        let mv_iq_tq_library = device
            .new_library_with_source(mv_iq_tq_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_iq_tq library: {}", e))?;

        // Create cached pipeline for Q4_0 MV
        let mv_q4_0_kernel = mv_q4_0_library.get_function("kernel_mul_mv_q4_0_f32", None)
            .map_err(|e| format!("Failed to get Q4_0 kernel: {}", e))?;
        let mv_q4_0_pipeline = device.new_compute_pipeline_state_with_function(&mv_q4_0_kernel)
            .map_err(|e| format!("Failed to create Q4_0 pipeline: {}", e))?;

        // Create cached pipeline for Q4_K MV
        let mv_q4_k_kernel = mv_q4_k_library.get_function("kernel_mul_mv_q4_K_f32", None)
            .map_err(|e| format!("Failed to get Q4_K kernel: {}", e))?;
        let mv_q4_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q4_k_kernel)
            .map_err(|e| format!("Failed to create Q4_K pipeline: {}", e))?;

        // Create cached pipeline for Q5_K MV
        let mv_q5_k_kernel = mv_q5_k_library.get_function("kernel_mul_mv_q5_K_f32", None)
            .map_err(|e| format!("Failed to get Q5_K kernel: {}", e))?;
        let mv_q5_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q5_k_kernel)
            .map_err(|e| format!("Failed to create Q5_K pipeline: {}", e))?;

        // Create cached pipeline for Q2_K MV
        let mv_q2_k_kernel = mv_q2_k_library.get_function("kernel_mul_mv_q2_K_f32", None)
            .map_err(|e| format!("Failed to get Q2_K kernel: {}", e))?;
        let mv_q2_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q2_k_kernel)
            .map_err(|e| format!("Failed to create Q2_K pipeline: {}", e))?;

        // Create cached pipeline for Q6_K MV
        let mv_q6_k_kernel = mv_q6_k_library.get_function("kernel_mul_mv_q6_K_f32", None)
            .map_err(|e| format!("Failed to get Q6_K kernel: {}", e))?;
        let mv_q6_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q6_k_kernel)
            .map_err(|e| format!("Failed to create Q6_K pipeline: {}", e))?;

        // Create cached pipeline for Q3_K MV
        let mv_q3_k_kernel = mv_q3_k_library.get_function("kernel_mul_mv_q3_K_f32", None)
            .map_err(|e| format!("Failed to get Q3_K kernel: {}", e))?;
        let mv_q3_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q3_k_kernel)
            .map_err(|e| format!("Failed to create Q3_K pipeline: {}", e))?;

        // Create cached pipeline for IQ4_NL MV
        let mv_iq4_nl_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq4_nl_f32", None)
            .map_err(|e| format!("Failed to get IQ4_NL kernel: {}", e))?;
        let mv_iq4_nl_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq4_nl_kernel)
            .map_err(|e| format!("Failed to create IQ4_NL pipeline: {}", e))?;

        Ok(Self {
            device,
            queue,
            vec_dot_library,
            mv_library,
            mv_simd_library,
            mv_q4_0_library,
            mv_q4_k_library,
            mv_q5_k_library,
            mv_q2_k_library,
            mv_q6_k_library,
            mv_q3_k_library,
            mv_iq_tq_library,
            mv_q4_0_pipeline,
            mv_q4_k_pipeline,
            mv_q5_k_pipeline,
            mv_q2_k_pipeline,
            mv_q6_k_pipeline,
            mv_q3_k_pipeline,
            mv_iq4_nl_pipeline,
            mv_weights_buffer: RefCell::new(None),
            mv_input_buffer: RefCell::new(None),
            mv_output_buffer: RefCell::new(None),
            mv_weights_size: RefCell::new(0),
            mv_input_size: RefCell::new(0),
            mv_output_size: RefCell::new(0),
        })
    }

    pub fn vec_dot_iq4_nl_q8_0(&self, n: usize, x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> Result<f32, String> {
        let nb = n / 32;

        let x_buffer = self.device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            (x.len() * std::mem::size_of::<BlockIQ4NL>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let y_buffer = self.device.new_buffer_with_data(
            y.as_ptr() as *const std::ffi::c_void,
            (y.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let kernel = self.vec_dot_library.get_function("vec_dot_iq4_nl_q8_0", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&y_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(nb as u32) as *const u32 as *const std::ffi::c_void);

        let thread_group_count = MTLSize { width: 1, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 1, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result_ptr = result_buffer.contents() as *const f32;
        Ok(unsafe { *result_ptr })
    }

    pub fn mv_iq4_nl_q8_0(&self, m: usize, k: usize, weights: &[BlockIQ4NL], input: &[BlockQ8_0]) -> Result<Vec<f32>, String> {
        let nb = k / 32;

        // Time buffer creation
        let t0 = std::time::Instant::now();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockIQ4NL>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (m * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let buffer_time = t0.elapsed();

        let t1 = std::time::Instant::now();

        let kernel = self.mv_library.get_function("mv_iq4_nl_q8_0", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let pipeline_time = t1.elapsed();

        let t2 = std::time::Instant::now();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(m as u32) as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(nb as u32) as *const u32 as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp
        // NSG = 2 SIMD groups per threadgroup, each SIMD group has 32 threads
        // Each SIMD group processes NR0 = 4 rows
        // Total: 2 * 32 = 64 threads, processing 2 * 4 = 8 rows per threadgroup
        const NSG: u64 = 2;
        const NR0: u64 = 4;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 8
        const THREADS_PER_THREADGROUP: u64 = NSG * 32; // 64

        let thread_group_size = MTLSize {
            width: THREADS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        let encode_time = t2.elapsed();

        let t3 = std::time::Instant::now();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let gpu_time = t3.elapsed();

        // Only log timing in debug builds
        #[cfg(debug_assertions)]
        eprintln!("MV timing: buffer={:?} pipeline={:?} encode={:?} gpu={:?}",
                  buffer_time, pipeline_time, encode_time, gpu_time);

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// SIMD-optimized MV kernel (32 threads per row)
    pub fn mv_iq4_nl_q8_0_simd(&self, m: usize, k: usize, weights: &[BlockIQ4NL], input: &[BlockQ8_0]) -> Result<Vec<f32>, String> {
        let nb = k / 32;

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockIQ4NL>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (m * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let kernel = self.mv_simd_library.get_function("mv_iq4_nl_q8_0_simd", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(m as u32) as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(nb as u32) as *const u32 as *const std::ffi::c_void);

        // One threadgroup per row (32 threads per threadgroup)
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count = MTLSize { width: m as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q4_0 x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q4_0_f32(&self, m: usize, k: usize, weights: &[BlockQ4_0], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64; // Byte stride per row
        const NR0: u64 = 4;

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ4_0>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Output buffer
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Use cached pipeline
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q4_0_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 8

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q4_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q4_k_f32(&self, m: usize, k: usize, weights: &[BlockQ4K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q4_K = 2

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Output buffer
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Use cached pipeline
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q4_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q5_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q5_k_f32(&self, m: usize, k: usize, weights: &[BlockQ5K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q5_K = 2

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ5K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Output buffer - create once, reuse
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q5_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q2_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q2_k_f32(&self, m: usize, k: usize, weights: &[BlockQ2K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q2_K = 2

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Output buffer - create once, reuse
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q2_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q6_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q6_k_f32(&self, m: usize, k: usize, weights: &[BlockQ6K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q6_K = 2

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q6_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q3_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q3_k_f32(&self, m: usize, k: usize, weights: &[BlockQ3K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ3K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q3_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ4_NL x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_iq4_nl_f32(&self, m: usize, k: usize, weights: &[BlockIQ4NL], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 32;  // QK4_NL = 32
        let nb01 = (nb * std::mem::size_of::<BlockIQ4NL>()) as u64;
        const NR0: u64 = 4;  // N_R0_IQ4_NL = 4

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockIQ4NL>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq4_nl_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }
}
