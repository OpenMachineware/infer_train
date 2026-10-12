// MoE GEMM correctness test - compare with reference MV kernel
// For 1 expert, 1 token, MoE kernel should match regular MV kernel

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
    println!("=== MoE GEMM Correctness Test ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let llama_lib = device
        .new_library_with_file("shaders/mul_mm_template.metallib")
        .expect("Failed to load llama.cpp library");
    let self_lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .expect("Failed to load self-written library");

    println!("✓ Libraries loaded\n");

    // Test with q4_0 (simpler block structure)
    let m = 64;
    let k = 128;
    let n_experts = 1;
    let n_tokens = 32; // Use 32 tokens to match kernel tile size
    let n_expert_used = 1;

    let cv = create_function_constants();

    let block_bytes = 18; // q4_0 block size
    let block_size = 32;
    let num_blocks = k / block_size;
    let NR1: usize = 32; // Kernel tile size for tokens

    println!(
        "Testing q4_0 (M={}, K={}, 1 expert, {} tokens)...",
        m, k, n_tokens
    );

    // Create weights for 1 expert
    let mut weights = vec![0u8; m * num_blocks * block_bytes];
    for row in 0..m {
        for block_idx in 0..num_blocks {
            let offset =
                row * num_blocks * block_bytes + block_idx * block_bytes;
            // block_q4_0: d (2 bytes) + qs (16 bytes)
            let d_bits = half::f16::from_f32(0.125).to_bits();
            weights[offset..offset + 2].copy_from_slice(&d_bits.to_le_bytes());
            // Simple 4-bit pattern
            for i in 0..16 {
                weights[offset + 2 + i] = (i as u8) | ((15 - i as u8) << 4);
            }
        }
    }

    // Create input (F32) for all tokens
    let input: Vec<f32> =
        (0..(k * n_tokens)).map(|i| 0.01 * ((i % k) as f32)).collect();

    // Create output buffers - ZERO initialized
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

    // Simple routing: 32 tokens assigned to expert 0
    let htpe = vec![n_tokens as u32]; // expert 0 processes n_tokens tokens
    let hids: Vec<i32> = (0..n_tokens as i32).collect(); // token IDs 0..n_tokens
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

    // Args for MoE kernel
    let row_stride = (num_blocks as u64) * (block_bytes as u64);
    let args = MulMMIdArgs {
        ne00: k as i32,
        ne02: n_experts as i32,
        nb01: row_stride,
        nb02: m as u64 * row_stride, // stride for expert dimension
        nb03: 0,
        ne11: k as i32,
        nb10: 4,            // element stride
        nb11: 4,            // row stride (1 element, column-major)
        nb12: 4 * k as u64, // column stride (K elements per token)
        nb13: 0,
        ne20: n_expert_used as i32,
        ne21: n_tokens as i32,
        ne0: m as i32,
        ne1: 1, // stride for token batches (1 since tokens are contiguous)
        r2: 1,
        r3: 1,
    };

    println!("Kernel args:");
    println!("  ne00={}, ne02={}", args.ne00, args.ne02);
    println!("  nb01={}, nb02={}", args.nb01, args.nb02);
    println!("  ne11={}", args.ne11);
    println!("  nb10={}, nb11={}", args.nb10, args.nb11);
    println!("  ne20={}, ne21={}", args.ne20, args.ne21);
    println!("  ne0={}, ne1={}", args.ne0, args.ne1);
    println!();

    let amax_data = vec![1.0f32, 1.0f32];
    let amax_buffer = device.new_buffer_with_data(
        amax_data.as_ptr() as *const c_void,
        8,
        MTLResourceOptions::StorageModeShared,
    );

    // Get MoE kernels
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

    // Run MoE kernels
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

        // For M=64, NR0=64, we need 1 threadgroup in y
        // For neh1=32, NR1=32, we need 1 threadgroup in x
        // For n_experts=1, we need 1 threadgroup in z
        let threadgroup_size = MTLSize::new(128, 1, 1);
        let num_threadgroups_x = (n_tokens + NR1 - 1) / NR1;
        let threadgroups = MTLSize::new(num_threadgroups_x as u64, 1, 1);

        println!(
            "{}: threadgroups=({},{},{}), threadgroup_size=({},{},{})",
            name,
            threadgroups.width,
            threadgroups.height,
            threadgroups.depth,
            threadgroup_size.width,
            threadgroup_size.height,
            threadgroup_size.depth
        );

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Compare MoE results
    let self_ptr = output_moe_self.contents() as *const f32;
    let llama_ptr = output_moe_llama.contents() as *const f32;
    let self_slice =
        unsafe { std::slice::from_raw_parts(self_ptr, output_size) };
    let llama_slice =
        unsafe { std::slice::from_raw_parts(llama_ptr, output_size) };

    println!("\nOutput comparison:");
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;
    let mut nan_count = 0;
    let mut large_diff_indices = vec![];

    for i in 0..output_size {
        if self_slice[i].is_nan() || llama_slice[i].is_nan() {
            nan_count += 1;
        }
        let diff = (self_slice[i] - llama_slice[i]).abs();
        if diff > 1e-4 {
            diff_count += 1;
            if diff > 1.0 {
                large_diff_indices.push((
                    i,
                    self_slice[i],
                    llama_slice[i],
                    diff,
                ));
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("  NaN count: {}", nan_count);
    println!("  Values with diff > 1e-4: {}", diff_count);
    println!("  Max diff: {:.6e}", max_diff);

    if !large_diff_indices.is_empty() {
        println!("\n  Large differences (diff > 1.0):");
        for (i, s, l, d) in large_diff_indices.iter().take(20) {
            println!(
                "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
                i, s, l, d
            );
        }
    }

    println!("\n  Token 0 (indices 0-63):");
    for i in 0..8 {
        println!(
            "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
            i,
            self_slice[i],
            llama_slice[i],
            (self_slice[i] - llama_slice[i]).abs()
        );
    }

    println!("\n  Token 1 (indices 64-127):");
    for i in 64..72 {
        println!(
            "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
            i,
            self_slice[i],
            llama_slice[i],
            (self_slice[i] - llama_slice[i]).abs()
        );
    }

    println!("\n  Token 2 (indices 128-191):");
    for i in 128..136 {
        println!(
            "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
            i,
            self_slice[i],
            llama_slice[i],
            (self_slice[i] - llama_slice[i]).abs()
        );
    }

    if max_diff < 1e-4 {
        println!("\n  ✓ PASS");
    } else {
        println!("\n  ✗ FAIL");
    }

    println!("\n=== Done ===");
}
