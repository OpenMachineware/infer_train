// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::*;
use std::ffi::c_void;

const QK_K: usize = 256;

#[repr(C)]
struct MulMVArgs {
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

fn create_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();

    unsafe {
        // Use the indices directly
        cv.set_constant_value_at_index(
            &nsg as *const _ as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const _ as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const _ as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Load our rewritten shader
    let my_shader = include_str!("../src/ops/metal/gemm/mul_mm.metal");

    // Load llama.cpp standalone shader
    let llama_shader = include_str!("/tmp/q4_k_standalone.metal");

    let compile_options = CompileOptions::new();

    // Try to compile both
    println!("Compiling self-written shader...");
    let my_lib =
        match device.new_library_with_source(my_shader, &compile_options) {
            Ok(lib) => lib,
            Err(e) => {
                println!("Self-written shader compilation failed: {}", e);
                return;
            }
        };

    println!("Compiling llama.cpp standalone shader...");
    let llama_lib =
        match device.new_library_with_source(llama_shader, &compile_options) {
            Ok(lib) => lib,
            Err(e) => {
                println!("llama.cpp shader compilation failed: {}", e);
                return;
            }
        };

    // Test Q4_K only (llama.cpp standalone shader only has q4_K)
    test_format(&device, &queue, &my_lib, &llama_lib, "q4_K", 144, 2, 2);
}

fn test_format(
    device: &Device,
    queue: &CommandQueue,
    my_lib: &Library,
    llama_lib: &Library,
    name: &str,
    block_size: usize,
    nsg: i16,
    nr0: i16,
) {
    println!("\n=== Testing {} ===", name);

    let m = 1024usize;
    let k = 4096usize;
    let nb = k / QK_K;

    // Generate test data
    let weights_size = m * nb * block_size;
    let weights_data: Vec<u8> =
        (0..weights_size).map(|i| ((i * 17 + 13) % 256) as u8).collect();
    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

    // Create buffers
    let weights_buf = device.new_buffer_with_data(
        weights_data.as_ptr() as *const _,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let llama_out = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let my_out = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    let args = MulMVArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: (nb * block_size) as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let kernel_name = format!("kernel_mul_mm_{}_f32", name);
    let cv = create_function_constants(nsg, 1, 1, 1);

    // Get functions
    let llama_func =
        match llama_lib.get_function(&kernel_name, Some(cv.clone())) {
            Ok(f) => f,
            Err(e) => {
                println!("Failed to get llama.cpp function: {}", e);
                return;
            }
        };

    let my_func = match my_lib.get_function(&kernel_name, Some(cv)) {
        Ok(f) => f,
        Err(e) => {
            println!("Failed to get self-written function: {}", e);
            return;
        }
    };

    // Create pipelines
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    let grid = MTLSize::new(
        ((m + (nsg as usize * nr0 as usize) - 1)
            / (nsg as usize * nr0 as usize)) as u64,
        1,
        1,
    );
    let tg = MTLSize::new((nsg as u64) * 32, 1, 1);

    // Run llama.cpp kernel
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&llama_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MulMVArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buf), 0);
    enc.set_buffer(2, Some(&input_buf), 0);
    enc.set_buffer(3, Some(&llama_out), 0);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run self-written kernel
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&my_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MulMVArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buf), 0);
    enc.set_buffer(2, Some(&input_buf), 0);
    enc.set_buffer(3, Some(&my_out), 0);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Compare outputs
    let llama_output = unsafe {
        std::slice::from_raw_parts(llama_out.contents() as *const f32, m)
    };
    let my_output = unsafe {
        std::slice::from_raw_parts(my_out.contents() as *const f32, m)
    };

    let mut correct = 0;
    let mut max_rel_err: f64 = 0.0;
    let mut max_abs_err: f64 = 0.0;

    for i in 0..m {
        let abs_err = (my_output[i] - llama_output[i]).abs();
        let rel_err = if llama_output[i].abs() > 1e-6 {
            abs_err / llama_output[i].abs()
        } else {
            abs_err
        };

        max_abs_err = max_abs_err.max(abs_err as f64);
        max_rel_err = max_rel_err.max(rel_err as f64);

        if rel_err < 0.01 {
            correct += 1;
        }
    }

    println!(
        "Correctness: {}/{} ({:.1}%)",
        correct,
        m,
        correct as f64 / m as f64 * 100.0
    );
    println!("Max absolute error: {:.6e}", max_abs_err);
    println!("Max relative error: {:.6e}", max_rel_err);

    // Show some sample outputs
    println!("Sample outputs:");
    for i in [0, 1, 2, 100, 500, 1000].iter() {
        let i = *i.min(&(m - 1));
        println!(
            "  [{}]: llama={:.6}, ours={:.6}, rel_err={:.6}",
            i,
            llama_output[i],
            my_output[i],
            if llama_output[i].abs() > 1e-6 {
                (my_output[i] - llama_output[i]).abs() / llama_output[i].abs()
            } else {
                (my_output[i] - llama_output[i]).abs()
            }
        );
    }
}
