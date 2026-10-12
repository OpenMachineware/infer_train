// MoE vs Regular GEMM comparison
// For 1 expert, all tokens to expert 0, MoE should match regular GEMM

use metal::*;
use std::ffi::c_void;

#[derive(Clone, Copy)]
#[repr(C)]
struct MulMMArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

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

const FC_MUL_MM: NSUInteger = 700;

fn create_fc_gemm() -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let bc_inp = false;
        let bc_out = false;
        let one: i16 = 1;
        cv.set_constant_value_at_index(
            &bc_inp as *const _ as *const c_void,
            MTLDataType::Bool,
            FC_MUL_MM + 0,
        );
        cv.set_constant_value_at_index(
            &bc_out as *const _ as *const c_void,
            MTLDataType::Bool,
            FC_MUL_MM + 1,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 2,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 3,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 4,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 5,
        );
    }
    cv
}

fn create_fc_moe() -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let bc_inp = false;
        let bc_out = false;
        let one: i16 = 1;
        let amax = false;
        cv.set_constant_value_at_index(
            &bc_inp as *const _ as *const c_void,
            MTLDataType::Bool,
            FC_MUL_MM + 0,
        );
        cv.set_constant_value_at_index(
            &bc_out as *const _ as *const c_void,
            MTLDataType::Bool,
            FC_MUL_MM + 1,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 2,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 3,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 4,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            MTLDataType::Short,
            FC_MUL_MM + 5,
        );
        cv.set_constant_value_at_index(
            &amax as *const _ as *const c_void,
            MTLDataType::Bool,
            FC_MUL_MM + 6,
        );
    }
    cv
}

fn main() {
    println!("=== MoE vs Regular GEMM Comparison ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .expect("Failed to load library");

    // Test parameters - use dimensions that match a partial tile
    let m = 64; // output rows
    let k = 256; // input dimension
    let n_tokens = 4; // number of tokens (partial tile: 4 < 32)
    let block_bytes = 18;
    let block_size = 32;
    let num_blocks = k / block_size;

    // Create weights - simple pattern for easy verification
    let mut weights = vec![0u8; m * num_blocks * block_bytes];
    for row in 0..m {
        for block_idx in 0..num_blocks {
            let offset =
                row * num_blocks * block_bytes + block_idx * block_bytes;
            // Scale = 0.125
            let d_bits = half::f16::from_f32(0.125).to_bits();
            weights[offset..offset + 2].copy_from_slice(&d_bits.to_le_bytes());
            // All quantized values = 8 (neutral value for symmetric quantization)
            for i in 0..16 {
                weights[offset + 2 + i] = 0x88; // 8 in both nibbles
            }
        }
    }

    // Create input (F32) for all tokens
    let input: Vec<f32> =
        (0..(k * n_tokens)).map(|i| 0.01 * ((i % k) as f32)).collect();

    // Create output buffers
    let output_size = m * n_tokens;
    let output_gemm = device.new_buffer(
        output_size as u64 * 4,
        MTLResourceOptions::StorageModeShared,
    );
    let output_moe = device.new_buffer(
        output_size as u64 * 4,
        MTLResourceOptions::StorageModeShared,
    );

    // Zero outputs
    unsafe {
        std::ptr::write_bytes(
            output_gemm.contents() as *mut f32,
            0,
            output_size,
        );
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

    // Run regular GEMM
    {
        let fc = create_fc_gemm();
        let kernel_name = "kernel_mul_mm_q4_0_f32";
        let function = lib
            .get_function(kernel_name, Some(fc))
            .expect("GEMM kernel not found");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create pipeline");

        let row_stride = (num_blocks as u64) * (block_bytes as u64);
        let args = MulMMArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: block_bytes as u64,
            nb01: row_stride,
            nb02: m as u64 * row_stride,
            ne10: k as i32,
            ne11: n_tokens as i32,
            ne12: 1,
            nb10: 4,
            nb11: 4 * k as u64,
            nb12: 4 * k as u64 * n_tokens as u64,
            ne0: m as i32,
            ne1: n_tokens as i32,
            r2: 1,
            r3: 1,
        };

        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMMArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_gemm), 0);
        encoder.set_threadgroup_memory_length(0, 8192);

        let threadgroup_size = MTLSize::new(64, 2, 1);
        let threadgroups = MTLSize::new(
            ((n_tokens + 31) / 32) as u64,
            ((m + 63) / 64) as u64,
            1,
        );
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Run MoE GEMM
    {
        let fc = create_fc_moe();
        let kernel_name = "kernel_mul_mm_id_q4_0_f32";
        let function = lib
            .get_function(kernel_name, Some(fc))
            .expect("MoE kernel not found");
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

        // MoE kernel uses 1D threadgroup (128, 1, 1) to get 4 simdgroups (sgitg=0,1,2,3)
        let threadgroup_size = MTLSize::new(128, 1, 1);
        let threadgroups = MTLSize::new(
            ((n_tokens + 31) / 32) as u64,
            ((m + 63) / 64) as u64,
            1,
        );
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Compare results
    let gemm_ptr = output_gemm.contents() as *const f32;
    let moe_ptr = output_moe.contents() as *const f32;
    let gemm_slice =
        unsafe { std::slice::from_raw_parts(gemm_ptr, output_size) };
    let moe_slice = unsafe { std::slice::from_raw_parts(moe_ptr, output_size) };

    println!("Comparing {} outputs...", output_size);

    // Print all rows for token 0
    println!("\nAll rows (token 0):");
    for row in 0..m {
        let diff = (gemm_slice[row] - moe_slice[row]).abs();
        if diff > 1e-4 {
            println!(
                "Row {:3}: GEMM={:9.4}, MoE={:9.4}, diff={:.4}",
                row, gemm_slice[row], moe_slice[row], diff
            );
        }
    }

    // Check other tokens
    for token in 1..n_tokens {
        let base = token * m;
        println!("\nToken {} first 5 rows:", token);
        for row in 0..5 {
            println!(
                "  Row {:3}: GEMM={:9.4}, MoE={:9.4}",
                row,
                gemm_slice[base + row],
                moe_slice[base + row]
            );
        }
    }

    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for token in 0..n_tokens {
        for row in 0..m {
            let idx = token * m + row;
            let diff = (gemm_slice[idx] - moe_slice[idx]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-4 {
                diff_count += 1;
            }
        }
    }

    println!("\nMax diff: {:.6}", max_diff);
    println!("Diff count: {}/{}", diff_count, output_size);

    if max_diff < 1e-4 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
