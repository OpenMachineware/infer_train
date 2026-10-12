// Simple MoE test with full tile (32 tokens)

use metal::*;
use std::ffi::c_void;

#[derive(Clone, Copy)]
#[repr(C)]
struct MulMMIdArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne20: i32,
    ne21: i32,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

fn main() {
    println!("=== Simple MoE Test with Full Tile ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_file("src/ops/metal/mul_mm_standalone.metallib")
        .expect("Failed to load library");

    // Test parameters - full tile (32 tokens)
    let m = 64;
    let k = 256;
    let n_tokens = 32; // Full tile
    let block_bytes = 18;
    let block_size = 32;
    let num_blocks = k / block_size;

    // Create weights - all zeros
    let mut weights = vec![0u8; m * num_blocks * block_bytes];
    for row in 0..m {
        for block_idx in 0..num_blocks {
            let offset =
                row * num_blocks * block_bytes + block_idx * block_bytes;
            let d_bits = half::f16::from_f32(0.125).to_bits();
            weights[offset..offset + 2].copy_from_slice(&d_bits.to_le_bytes());
            for i in 0..16 {
                weights[offset + 2 + i] = 0x88;
            }
        }
    }

    // Create input
    let input: Vec<f32> =
        (0..(k * n_tokens)).map(|i| 0.01 * ((i % k) as f32)).collect();

    // Create output
    let output_size = m * n_tokens;
    let output_moe = device.new_buffer(
        output_size as u64 * 4,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::write_bytes(
            output_moe.contents() as *mut f32,
            0,
            output_size,
        );
    }

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // MoE kernel
    let cv = FunctionConstantValues::new();
    unsafe {
        let bc_inp = false;
        let bc_out = false;
        let one: i16 = 1;
        let amax = false;
        cv.set_constant_value_at_index(
            &bc_inp as *const _ as *const c_void,
            MTLDataType::Bool,
            700,
        );
        cv.set_constant_value_at_index(
            &bc_out as *const _ as *const c_void,
            MTLDataType::Bool,
            701,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            702,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            703,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            704,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            705,
        );
        cv.set_constant_value_at_index(
            &amax as *const _ as *const c_void,
            MTLDataType::Bool,
            706,
        );
    }

    let kernel_name = "kernel_mul_mm_id_q4_0_f32";
    let function =
        lib.get_function(kernel_name, Some(cv)).expect("MoE kernel not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline");

    // Routing: all tokens to expert 0
    let htpe = vec![n_tokens as u32];
    let hids: Vec<i32> = (0..n_tokens as i32).collect();
    let htpe_buffer = device.new_buffer_with_data(
        htpe.as_ptr() as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let hids_buffer = device.new_buffer_with_data(
        hids.as_ptr() as *const c_void,
        (hids.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let amax_data = vec![1.0f32, 1.0f32];
    let amax_buffer = device.new_buffer_with_data(
        amax_data.as_ptr() as *const c_void,
        8,
        MTLResourceOptions::StorageModeShared,
    );

    let row_stride = (num_blocks as u64) * (block_bytes as u64);
    let args = MulMMIdArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: row_stride,
        nb02: m as u64 * row_stride,
        nb03: 0,
        ne11: k as i32,
        nb10: 4,
        nb11: 4,
        nb12: 4 * k as u64,
        nb13: 0,
        ne20: 1,
        ne21: n_tokens as i32,
        ne0: m as i32,
        ne1: 1,
        r2: 1,
        r3: 1,
    };

    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(
        0,
        std::mem::size_of::<MulMMIdArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&htpe_buffer), 0);
    encoder.set_buffer(4, Some(&hids_buffer), 0);
    encoder.set_buffer(5, Some(&output_moe), 0);
    encoder.set_buffer(6, Some(&amax_buffer), 0);
    encoder.set_threadgroup_memory_length(0, 8192);

    // Use 1D threadgroup (128, 1, 1) for 4 simdgroups
    let threadgroup_size = MTLSize::new(128, 1, 1);
    let threadgroups = MTLSize::new(1, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Check results
    let moe_ptr = output_moe.contents() as *const f32;
    let moe_slice = unsafe { std::slice::from_raw_parts(moe_ptr, output_size) };

    println!("First 10 rows (token 0):");
    for row in 0..10 {
        println!("Row {:3}: {:.6}", row, moe_slice[row]);
    }

    println!("\nFirst 10 rows (token 1):");
    for row in 0..10 {
        println!("Row {:3}: {:.6}", row, moe_slice[1 * m + row]);
    }

    println!("\nFirst 10 rows (token 15):");
    for row in 0..10 {
        println!("Row {:3}: {:.6}", row, moe_slice[15 * m + row]);
    }

    // Check for non-zero values
    let mut non_zero_count = 0;
    for (i, &v) in moe_slice.iter().enumerate() {
        if v.abs() > 1e-6 {
            non_zero_count += 1;
            if non_zero_count <= 20 {
                println!("Non-zero at index {}: {:.6}", i, v);
            }
        }
    }

    println!("\nNon-zero count: {}/{}", non_zero_count, output_size);

    if non_zero_count == 0 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
