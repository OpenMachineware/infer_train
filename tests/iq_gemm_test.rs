// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// IQ GEMM correctness test
use metal::*;
use std::ffi::c_void;

#[derive(Clone, Copy)]
#[repr(C)]
struct MulMMArgs {
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
    ne0: i32,
    ne1: i32,
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
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let self_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_standalone.metallib")
        .unwrap();
    let llama_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_template.metallib")
        .unwrap();

    let iq_kernels = [
        "kernel_mul_mm_iq4_nl_f32",
        "kernel_mul_mm_iq4_xs_f32",
        "kernel_mul_mm_iq3_s_f32",
        "kernel_mul_mm_iq3_xxs_f32",
        "kernel_mul_mm_iq2_s_f32",
        "kernel_mul_mm_iq2_xs_f32",
        "kernel_mul_mm_iq2_xxs_f32",
        "kernel_mul_mm_iq1_s_f32",
        "kernel_mul_mm_iq1_m_f32",
    ];

    println!("=== IQ GEMM Correctness Test ===");
    println!("M=64, K=256, N=64\n");

    for kernel in iq_kernels {
        let cv = create_function_constants();

        // Try to get the function from both libraries
        let self_fn = match self_lib.get_function(kernel, Some(cv.clone())) {
            Ok(f) => f,
            Err(e) => {
                println!("{}: ✗ Self kernel not found: {}", kernel, e);
                continue;
            }
        };

        let llama_fn = match llama_lib.get_function(kernel, Some(cv.clone())) {
            Ok(f) => f,
            Err(e) => {
                println!("{}: ✗ Llama kernel not found: {}", kernel, e);
                continue;
            }
        };

        let self_pipe =
            device.new_compute_pipeline_state_with_function(&self_fn).unwrap();
        let llama_pipe =
            device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

        // Simple test data
        let m = 64;
        let k = 256;
        let n = 64;
        let block_size = 256; // QK_K
        let num_blocks = k / block_size;
        let block_bytes = 256 + 2; // Approximate size for IQ formats

        let weights = vec![0u8; m * num_blocks * block_bytes];
        let input: Vec<f32> =
            (0..(k * n)).map(|i| 0.01 * ((i % k) as f32)).collect();
        let output_size = m * n;

        let output_self = device.new_buffer(
            output_size as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );
        let output_llama = device.new_buffer(
            output_size as u64 * 4,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::write_bytes(
                output_self.contents() as *mut f32,
                0,
                output_size,
            );
            std::ptr::write_bytes(
                output_llama.contents() as *mut f32,
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

        let args = MulMMArgs {
            ne00: k as i32,
            ne02: 1,
            nb01: block_bytes as u64,
            nb02: m as u64 * block_bytes as u64,
            nb03: 0,
            ne11: k as i32,
            nb10: 4,
            nb11: 4,
            nb12: 4 * k as u64,
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
        };

        for (pipe, output) in [&self_pipe, &llama_pipe]
            .iter()
            .zip([&output_self, &output_llama].iter())
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<MulMMArgs>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(output), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(
                MTLSize::new(2, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let self_ptr = output_self.contents() as *const f32;
        let llama_ptr = output_llama.contents() as *const f32;
        let self_slice =
            unsafe { std::slice::from_raw_parts(self_ptr, output_size) };
        let llama_slice =
            unsafe { std::slice::from_raw_parts(llama_ptr, output_size) };

        let mut max_diff = 0.0f32;
        let mut nan_count = 0;
        for i in 0..output_size {
            if self_slice[i].is_nan() || llama_slice[i].is_nan() {
                nan_count += 1;
            }
            let diff = (self_slice[i] - llama_slice[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        if nan_count > 0 {
            println!("{}: ✗ NaN count={}", kernel, nan_count);
        } else if max_diff < 1e-3 {
            println!("{}: ✓ PASS (max_diff={:.6})", kernel, max_diff);
        } else {
            println!("{}: ✗ FAIL (max_diff={:.2})", kernel, max_diff);
        }
    }
}
