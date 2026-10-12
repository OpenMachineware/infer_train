use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLSize,
};
use std::cell::RefCell;

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

impl FlashAttnMetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        // Combine dequantize.h and flash_attn.metal
        let dequantize_source = include_str!("include/dequantize.h");
        let flash_attn_source = include_str!("attention/flash_attn.metal");
        // Remove the include directive and concatenate
        let source = format!(
            "{}\n{}",
            dequantize_source,
            flash_attn_source.replace("#include \"dequantize.h\"", "")
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
