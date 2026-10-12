// Simple MV test to verify framework
use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use std::ffi::c_void;

fn main() {
    println!("=== MV Simple Test ===");
    let device = Device::system_default().expect("No Metal device");

    // Load gemm.metal (MV kernel) at runtime
    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/blocks.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../src/ops/metal/iq_grid_tables.h");

    let combined =
        format!("{}\n{}\n{}", ggml_common_metal, iq_grid_tables, gemm_source);
    let compile_opts = metal::CompileOptions::new();
    let my_lib = device
        .new_library_with_source(&combined, &compile_opts)
        .expect("Failed to compile gemm.metal");

    // Test Q4_0 MV: M=1, K=32
    let mut q4_0_block =
        BlockQ4_0 { d: f16::from_f32(1.0).to_bits(), qs: [0u8; 16] };
    for i in 0..16 {
        q4_0_block.qs[i] = (i | (i << 4)) as u8;
    }
    let input_f16: Vec<u16> = vec![f16::from_f32(1.0).to_bits(); 32];

    println!("Block: d=1.0, qs=[0..15] (low=high)");
    println!("Input: 32 ones");
    println!("Expected: sum((i-8) for i in 0..15) * 2 = 0");

    // MV args
    #[repr(C)]
    struct MvArgs {
        ne00: i32,
        ne01: i32,
        ne02: i32,
        nb00: u64,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        nr0: i32,
        r2: i16,
        r3: i16,
    }

    let args = MvArgs {
        ne00: 32,
        ne01: 1,
        ne02: 1,
        nb00: 1,
        nb01: 18,
        nb02: 0,
        nb03: 0,
        ne10: 32,
        ne11: 1,
        ne12: 1,
        nb10: 2,
        nb11: 64,
        nb12: 0,
        nb13: 0,
        ne0: 1,
        ne1: 1,
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let queue = device.new_command_queue();

    // Function constants for MV kernel
    const FC_MUL_MV: usize = 600;
    let cv = metal::FunctionConstantValues::new();
    let nsg_val: i16 = 4;
    let one_val: i16 = 1;
    cv.set_constant_value_at_index(
        &nsg_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        FC_MUL_MV as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 2) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 3) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 4) as u64,
    );

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &my_lib.get_function("kernel_mul_mm_q4_0_f32", Some(cv)).unwrap(),
        )
        .unwrap();

    let weights_arr = [q4_0_block];
    let weights_bytes: &[u8] = bytemuck::cast_slice(&weights_arr);
    let weights_buffer = device.new_buffer_with_data(
        weights_bytes.as_ptr() as *const c_void,
        weights_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        64u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output =
        device.new_buffer(4u64, metal::MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&output), 0);
    enc.set_threadgroup_memory_length(0, 8192);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = unsafe { *(output.contents() as *const f32) };
    println!("Result: {}", result);
    println!("Diff from expected: {}", (result - 0.0).abs());
}
