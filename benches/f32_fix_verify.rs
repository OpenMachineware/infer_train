// Verify F32 kernel fix - small safe test
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

#[repr(C)]
struct GemmArgsMm {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

fn main() {
    println!("=== F32 Kernel Fix Verification (Safe Small Test) ===");

    let device = Device::system_default().expect("No Metal device");
    let my_lib = device
        .new_library_with_file(std::path::Path::new(
            "src/ops/metal/mul_mm_standalone.metallib",
        ))
        .unwrap();
    let llama_lib = device
        .new_library_with_file(std::path::Path::new(
            "src/ops/metal/mul_mm_template.metallib",
        ))
        .unwrap();

    // Small safe sizes
    let m = 64usize;
    let k = 256usize;
    let n = 1usize;

    println!("Testing M={}, K={}, N={}", m, k, n);

    let queue = device.new_command_queue();
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    let weights_f32: Vec<f32> =
        (0..m * k).map(|_| rng.r#gen::<f32>() * 0.1).collect();
    let input_f16: Vec<u16> = (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect();

    const FC_MUL_MM: usize = 700;
    let true_val: bool = true;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &true_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM as u64,
        );
        cv.set_constant_value_at_index(
            &true_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            (FC_MUL_MM + 1) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 2) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 3) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 4) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 5) as u64,
        );
    }

    let my_fn =
        my_lib.get_function("kernel_mul_mm_f32_f32", Some(cv.clone())).unwrap();
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_f32_f32", Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights_f32.as_ptr() as *const c_void,
        (weights_f32.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_self = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (k * 4) as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    let nsg = 4;
    let nr0 = 1;
    let rows_per_tg = nsg * nr0;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((m + rows_per_tg - 1) / rows_per_tg) as u64, n as u64, 1);

    // Run self (no warmup, single run)
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    let max_diff = slice_self
        .iter()
        .zip(slice_llama.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\nResults:");
    println!("  Max diff: {:.6}", max_diff);
    if max_diff < 0.01 {
        println!("  ✓ PASS - Fix verified!");
    } else {
        println!("  ✗ FAIL - Still has issues");
        for i in 0..5.min(m * n) {
            println!(
                "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
                i,
                slice_self[i],
                slice_llama[i],
                (slice_self[i] - slice_llama[i]).abs()
            );
        }
    }
}
