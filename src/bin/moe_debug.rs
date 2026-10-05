// Debug MoE kernel memory layout
// Write specific values to threadgroup memory and verify where they end up

use metal::*;
use std::ffi::c_void;

fn main() {
    println!("=== MoE Kernel Memory Layout Debug ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Use self-written library
    let lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .expect("Failed to load library");

    // Test parameters - 4 tokens
    let m = 64;
    let k = 256;
    let n_tokens = 4;
    let block_bytes = 18;
    let block_size = 32;
    let num_blocks = k / block_size;

    // Create weights - use specific pattern to make debugging easier
    let mut weights = vec![0u8; m * num_blocks * block_bytes];
    for row in 0..m {
        for block_idx in 0..num_blocks {
            let offset =
                row * num_blocks * block_bytes + block_idx * block_bytes;
            // Scale = 1.0 for easier calculation
            let d_bits = half::f16::from_f32(1.0).to_bits();
            weights[offset..offset + 2].copy_from_slice(&d_bits.to_le_bytes());
            // Use row number as quantized value: row%16 for each nibble
            let val = (row % 16) as u8;
            for i in 0..16 {
                weights[offset + 2 + i] = val | (val << 4);
            }
        }
    }

    // Create input with specific pattern
    let input: Vec<f32> = (0..(k * n_tokens))
        .map(|i| {
            let token = i / k;
            let elem = i % k;
            (token + 1) as f32 * 0.01 * (elem as f32 + 1.0)
        })
        .collect();

    // Print input pattern
    println!("Input pattern:");
    for token in 0..n_tokens {
        println!(
            "  Token {}: first 4 elements = {:?}",
            token,
            &input[token * k..token * k + 4]
        );
    }

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

    // MoE kernel setup
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

    // Args
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

    let threadgroup_size = MTLSize::new(128, 1, 1);
    let threadgroups = MTLSize::new(1, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Check results
    let moe_ptr = output_moe.contents() as *const f32;
    let moe_slice = unsafe { std::slice::from_raw_parts(moe_ptr, output_size) };

    println!("\nOutput results:");
    for token in 0..n_tokens {
        println!(
            "  Token {} first 4 rows: {:.4} {:.4} {:.4} {:.4}",
            token,
            moe_slice[token * m],
            moe_slice[token * m + 1],
            moe_slice[token * m + 2],
            moe_slice[token * m + 3]
        );
    }

    // Check for NaN/Inf
    let mut nan_count = 0;
    let mut inf_count = 0;
    for &v in moe_slice.iter() {
        if v.is_nan() {
            nan_count += 1;
        }
        if v.is_infinite() {
            inf_count += 1;
        }
    }

    println!("\nNaN count: {}, Inf count: {}", nan_count, inf_count);

    if nan_count == 0 && inf_count == 0 {
        println!("✓ No NaN/Inf values");
    } else {
        println!("✗ Has NaN/Inf values");
    }
}
