// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Verify CONV_2D kernel correctness
use metal::{Device, MTLSize};

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/conv_2d.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Failed to compile");
    let kernel =
        library.get_function("conv_2d_f32_f32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    // Test: 1x1x4x4 input, 1x1x3x3 kernel, stride 1, no padding → 1x1x2x2 output
    let batch_sz: i32 = 1;
    let in_ch: i32 = 1;
    let out_ch: i32 = 1;
    let in_h: i32 = 4;
    let in_w: i32 = 4;
    let ker_h: i32 = 3;
    let ker_w: i32 = 3;
    let stride_val: i32 = 1;
    let dilation_val: i32 = 1;
    let pad_val: i32 = 0;
    let out_h = (in_h - ker_h + 2 * pad_val) / stride_val + 1;
    let out_w = (in_w - ker_w + 2 * pad_val) / stride_val + 1;

    // Input: 4x4 = [0..15]
    let input_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // Weights: 3x3 = [1,0,0, 0,1,0, 0,0,1] (identity-like)
    let weights: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    // CPU reference: simple convolution
    let mut cpu_out = vec![0.0f32; (out_h * out_w) as usize];
    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut acc = 0.0f32;
            for kh in 0..ker_h {
                let in_y = oh * stride_val + kh - pad_val;
                if in_y < 0 || in_y >= in_h {
                    continue;
                }
                for kw in 0..ker_w {
                    let in_x = ow * stride_val + kw - pad_val;
                    if in_x < 0 || in_x >= in_w {
                        continue;
                    }
                    let w = weights[(kh * ker_w + kw) as usize];
                    let x = input_data[(in_y * in_w + in_x) as usize];
                    acc += x * w;
                }
            }
            cpu_out[(oh * out_w + ow) as usize] = acc;
        }
    }

    // GPU execution
    let input_buf = device.new_buffer(
        (input_data.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let weight_buf = device.new_buffer(
        (weights.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (cpu_out.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input_data.as_ptr(),
            input_buf.contents() as *mut f32,
            input_data.len(),
        );
        std::ptr::copy_nonoverlapping(
            weights.as_ptr(),
            weight_buf.contents() as *mut f32,
            weights.len(),
        );
    }

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&weight_buf), 0);
    enc.set_buffer(1, Some(&input_buf), 0);
    enc.set_buffer(2, Some(&output_buf), 0);
    enc.set_bytes(3, 4, &batch_sz as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &in_ch as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &out_ch as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &in_h as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &in_w as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &out_h as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &out_w as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &ker_h as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &ker_w as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(12, 4, &stride_val as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(13, 4, &stride_val as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(
        14,
        4,
        &dilation_val as *const i32 as *const std::ffi::c_void,
    );
    enc.set_bytes(
        15,
        4,
        &dilation_val as *const i32 as *const std::ffi::c_void,
    );
    enc.set_bytes(16, 4, &pad_val as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(17, 4, &pad_val as *const i32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        MTLSize { width: cpu_out.len() as u64, height: 1, depth: 1 },
        MTLSize { width: 64, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_out = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const f32,
            cpu_out.len(),
        )
    };

    println!("=== CONV_2D ===");
    println!(
        "Input: {}x{}, Kernel: {}x{}, Stride: {}",
        in_h, in_w, ker_h, ker_w, stride_val
    );
    println!("Output: {}x{}", out_h, out_w);
    println!("Expected: {:?}", cpu_out);
    println!("Got:      {:?}", gpu_out);

    let max_diff = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {}", max_diff);
    if max_diff < 1e-5 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
