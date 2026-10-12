use half::f16;
use metal::{ComputePipelineState, Device, MTLLibrary, MTLSize};
use std::ffi::c_void;

// Block structures from types.rs
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ1S {
    pub d: u16,       // FP16 scale
    pub qs: [u8; 32], // Grid indices (11-bit each packed)
    pub qh: [u16; 8], // High bits
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ1M {
    pub qs: [u8; 32],    // Grid indices, low 8 bits
    pub qh: [u8; 16],    // Grid indices, high 3 bits + grid shift bit
    pub scales: [u8; 8], // 3-bit block scales
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ2XXS {
    pub d: u16,        // FP16 scale
    pub qs: [u16; 32], // Grid indices (8-bit each)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ2XS {
    pub d: u16,          // FP16 scale
    pub qs: [u16; 32],   // Grid indices (9-bit each packed)
    pub scales: [u8; 8], // 3-bit scales
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ3XXS {
    pub d: u16,       // FP16 scale
    pub qs: [u8; 96], // 3-bit indices packed
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GemmArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: i64,
    pub nb01: i64,
    pub nb02: i64,
    pub nb03: i64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: i64,
    pub nb11: i64,
    pub nb12: i64,
    pub nb13: i64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: i64,
    pub nb1: i64,
    pub nb2: i64,
    pub nb3: i64,
    pub r2: i32,
    pub r3: i32,
}

fn create_test_iq1s_blocks(n_blocks: usize, pattern: u8) -> Vec<BlockIQ1S> {
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        // Create block with varied indices
        // Use pattern to create different grid indices
        let mut qs = [0u8; 32];
        let mut qh = [0u16; 8];
        for j in 0..32 {
            qs[j] = (pattern as u32 * (i * 32 + j) as u32) as u8;
        }
        for j in 0..8 {
            qh[j] = (pattern as u32 * (i * 8 + j) as u32) as u16;
        }
        blocks.push(BlockIQ1S {
            d: half::f16::from_f32(1.0).to_bits(),
            qs,
            qh,
        });
    }
    blocks
}

fn create_test_iq2xxs_blocks(
    n_blocks: usize,
    pattern: u16,
) -> Vec<BlockIQ2XXS> {
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let mut qs = [0u16; 32];
        for j in 0..32 {
            qs[j] = (pattern.wrapping_mul((i * 32 + j) as u16)) & 0xFF; // 8-bit index
        }
        blocks.push(BlockIQ2XXS { d: half::f16::from_f32(1.0).to_bits(), qs });
    }
    blocks
}

fn create_test_iq2xs_blocks(n_blocks: usize, pattern: u16) -> Vec<BlockIQ2XS> {
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let mut qs = [0u16; 32];
        let mut scales = [0u8; 8];
        for j in 0..32 {
            qs[j] = (pattern.wrapping_mul((i * 32 + j) as u16)) & 0x1FF; // 9-bit index
        }
        for j in 0..8 {
            scales[j] = (pattern as u8).wrapping_mul((i * 8 + j) as u8) & 0x07; // 3-bit scale
        }
        blocks.push(BlockIQ2XS {
            d: half::f16::from_f32(1.0).to_bits(),
            qs,
            scales,
        });
    }
    blocks
}

fn create_test_iq3xxs_blocks(n_blocks: usize, pattern: u8) -> Vec<BlockIQ3XXS> {
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let mut qs = [0u8; 96];
        for j in 0..96 {
            qs[j] = pattern.wrapping_mul((i * 96 + j) as u8);
        }
        blocks.push(BlockIQ3XXS { d: half::f16::from_f32(1.0).to_bits(), qs });
    }
    blocks
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load libraries
    let my_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_standalone.metallib")
        .unwrap();
    let llama_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_template.metallib")
        .unwrap();

    // Test parameters
    let m = 64; // rows
    let k = 256; // cols
    let n = 64; // output cols

    // Create input matrix (FP16)
    let input_f16: Vec<f16> =
        (0..(k * n)).map(|i| f16::from_f32((i % 10) as f32 * 0.1)).collect();

    // Test IQ1_S
    println!("=== Testing IQ1_S ===");
    test_iq_format::<BlockIQ1S>(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq1_s_f16",
        "kernel_mul_mm_iq1_s_f16",
        &create_test_iq1s_blocks(m * k / 256, 0x55), // alternating pattern
        &input_f16,
        m,
        k,
        n,
    );

    // Test IQ2_XXS
    println!("\n=== Testing IQ2_XXS ===");
    test_iq_format::<BlockIQ2XXS>(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq2_xxs_f16",
        "kernel_mul_mm_iq2_xxs_f16",
        &create_test_iq2xxs_blocks(m * k / 256, 0xAA), // alternating pattern
        &input_f16,
        m,
        k,
        n,
    );

    // Test IQ2_XS
    println!("\n=== Testing IQ2_XS ===");
    test_iq_format::<BlockIQ2XS>(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq2_xs_f16",
        "kernel_mul_mm_iq2_xs_f16",
        &create_test_iq2xs_blocks(m * k / 256, 0x123), // varied pattern
        &input_f16,
        m,
        k,
        n,
    );

    // Test IQ3_XXS
    println!("\n=== Testing IQ3_XXS ===");
    test_iq_format::<BlockIQ3XXS>(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq3_xxs_f16",
        "kernel_mul_mm_iq3_xxs_f16",
        &create_test_iq3xxs_blocks(m * k / 256, 0x7), // 3-bit pattern
        &input_f16,
        m,
        k,
        n,
    );
}

fn test_iq_format<B: bytemuck::Pod + bytemuck::Zeroable + Copy>(
    device: &Device,
    queue: &metal::CommandQueue,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    my_kernel_name: &str,
    llama_kernel_name: &str,
    weights: &[B],
    input_f16: &[f16],
    m: usize,
    k: usize,
    n: usize,
) {
    // Function constants for mul_mm kernels
    let cv = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    let true_val: bool = true;
    let one_val: i16 = 1;

    // FC_MUL_MM = 700
    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        700,
    ); // bc_inp
    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        701,
    ); // bc_out
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        702,
    ); // ne12
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        703,
    ); // ne13
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        704,
    ); // r2
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        705,
    ); // r3
    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        706,
    ); // id_amax

    let my_pipeline = device
        .new_compute_pipeline_state_with_function(
            &my_lib.get_function(my_kernel_name, Some(cv.clone())).unwrap(),
        )
        .unwrap();
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib.get_function(llama_kernel_name, Some(cv)).unwrap(),
        )
        .unwrap();

    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        ne03: 1,
        nb00: std::mem::size_of::<B>() as i64,
        nb01: (k / 256 * std::mem::size_of::<B>()) as i64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        ne13: 1,
        nb10: 2,
        nb11: (k * 2) as i64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        ne2: 1,
        ne3: 1,
        nb0: 4,
        nb1: (m * 4) as i64,
        nb2: 0,
        nb3: 0,
        r2: 1,
        r3: 1,
    };

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<B>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_my = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(1, 1, 1);

    // Run my kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_my), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare results
    let my_ptr = output_my.contents() as *const f32;
    let llama_ptr = output_llama.contents() as *const f32;
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for i in 0..(m * n) {
        let my_val = unsafe { *my_ptr.add(i) };
        let llama_val = unsafe { *llama_ptr.add(i) };
        let diff = (my_val - llama_val).abs();
        if diff > 0.001 {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    if diff_count == 0 {
        println!("✓ PASS - All {} outputs match", m * n);
    } else {
        println!(
            "✗ FAIL - {}/{} differ, max_diff = {}",
            diff_count,
            m * n,
            max_diff
        );
        // Print first few differences
        for i in 0..(m * n).min(10) {
            let my_val = unsafe { *my_ptr.add(i) };
            let llama_val = unsafe { *llama_ptr.add(i) };
            if (my_val - llama_val).abs() > 0.001 {
                println!(
                    "  [{}] my={}, llama={}, diff={}",
                    i,
                    my_val,
                    llama_val,
                    (my_val - llama_val).abs()
                );
            }
        }
    }
}
