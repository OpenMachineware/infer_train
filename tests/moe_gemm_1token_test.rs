// MoE GEMM correctness test - 1 token case
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

const FC_MUL_MM: NSUInteger = 700;

fn create_function_constants() -> FunctionConstantValues {
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
    println!("=== MoE GEMM Correctness Test (1 Token) ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let llama_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_template.metallib")
        .expect("Failed to load llama.cpp library");
    let self_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_standalone.metallib")
        .expect("Failed to load self-written library");

    // Test with q4_0
    let m = 64;
    let k = 128;
    let n_experts = 1;
    let n_tokens = 1; // Only 1 token
    let n_expert_used = 1;

    let cv = create_function_constants();

    let block_bytes = 18; // q4_0 block size
    let block_size = 32;
    let num_blocks = k / block_size;

    println!("Testing q4_0 (M={}, K={}, 1 expert, 1 token)...", m, k);

    // Create weights for 1 expert
    let mut weights = vec![0u8; m * num_blocks * block_bytes];
    for row in 0..m {
        for block_idx in 0..num_blocks {
            let offset =
                row * num_blocks * block_bytes + block_idx * block_bytes;
            let d_bits = half::f16::from_f32(0.125).to_bits();
            weights[offset..offset + 2].copy_from_slice(&d_bits.to_le_bytes());
            for i in 0..16 {
                weights[offset + 2 + i] = (i as u8) | ((15 - i as u8) << 4);
            }
        }
    }

    // Create input (F32) for 1 token
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32)).collect();

    // Create output buffers
    let output_size = m * n_tokens * n_expert_used;
    let output_moe_self = {
        let buf = device.new_buffer(
            output_size as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );
        let ptr = buf.contents() as *mut f32;
        unsafe {
            std::ptr::write_bytes(ptr, 0, output_size);
        }
        buf
    };
    let output_moe_llama = {
        let buf = device.new_buffer(
            output_size as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );
        let ptr = buf.contents() as *mut f32;
        unsafe {
            std::ptr::write_bytes(ptr, 0, output_size);
        }
        buf
    };

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

    // Routing: 1 token to expert 0
    let htpe = vec![n_tokens as u32];
    let hids = vec![0i32];
    let htpe_buffer = device.new_buffer_with_data(
        htpe.as_ptr() as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let hids_buffer = device.new_buffer_with_data(
        hids.as_ptr() as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Args
    let row_stride = (num_blocks as u64) * (block_bytes as u64);
    let args = MulMMIdArgs {
        ne00: k as i32,
        ne02: n_experts as i32,
        nb01: row_stride,
        nb02: m as u64 * row_stride,
        nb03: 0,
        ne11: k as i32,
        nb10: 4,
        nb11: 4,
        nb12: 4 * k as u64,
        nb13: 0,
        ne20: n_expert_used as i32,
        ne21: n_tokens as i32,
        ne0: m as i32,
        ne1: 1,
        r2: 1,
        r3: 1,
    };

    let amax_data = vec![1.0f32, 1.0f32];
    let amax_buffer = device.new_buffer_with_data(
        amax_data.as_ptr() as *const c_void,
        8,
        MTLResourceOptions::StorageModeShared,
    );

    // Get kernels
    let kernel_name = "kernel_mul_mm_id_q4_0_f32";
    let self_function = self_lib
        .get_function(kernel_name, Some(cv.clone()))
        .expect("Self MoE kernel not found");
    let llama_function = llama_lib
        .get_function(kernel_name, Some(cv.clone()))
        .expect("Llama MoE kernel not found");

    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_function)
        .expect("Failed to create self pipeline");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_function)
        .expect("Failed to create llama pipeline");

    // Run kernels
    for (name, pipeline, output) in [
        ("self", &self_pipeline, &output_moe_self),
        ("llama", &llama_pipeline, &output_moe_llama),
    ] {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMMIdArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&htpe_buffer), 0);
        encoder.set_buffer(4, Some(&hids_buffer), 0);
        encoder.set_buffer(5, Some(output), 0);
        encoder.set_buffer(6, Some(&amax_buffer), 0);
        encoder.set_threadgroup_memory_length(0, 8192);

        let threadgroup_size = MTLSize::new(128, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Compare results
    let self_ptr = output_moe_self.contents() as *const f32;
    let llama_ptr = output_moe_llama.contents() as *const f32;
    let self_slice =
        unsafe { std::slice::from_raw_parts(self_ptr, output_size) };
    let llama_slice =
        unsafe { std::slice::from_raw_parts(llama_ptr, output_size) };

    println!("\nOutput comparison:");
    let mut max_diff = 0.0f32;
    for token in 0..n_tokens {
        println!(
            "\nToken {} (indices {}..{}):",
            token,
            token * m,
            (token + 1) * m - 1
        );
        for i in 0..8.min(m) {
            let idx = token * m + i;
            let diff = (self_slice[idx] - llama_slice[idx]).abs();
            max_diff = max_diff.max(diff);
            if diff > 1e-4 {
                println!(
                    "  [{}] self={:.6}, llama={:.6}, diff={:.6}",
                    idx, self_slice[idx], llama_slice[idx], diff
                );
            }
        }
    }

    println!("\nMax diff: {:.6}", max_diff);
    if max_diff < 1e-4 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }

    println!("\n=== Done ===");
}
