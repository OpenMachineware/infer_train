// Debug NaN pattern - is it first 64 rows?
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

const QK_K: usize = 256;

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

fn gen_iq2_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2XXS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u16; 32];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u16>();
            }
            BlockIQ2XXS { d: f16::from_f32(0.01).to_bits(), qs }
        })
        .collect()
}

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
    println!("=== NaN Pattern Debug ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let my_lib = device
        .new_library_with_file(std::path::Path::new(
            "shaders/mul_mm_standalone.metallib",
        ))
        .unwrap();
    let llama_lib = device
        .new_library_with_file(std::path::Path::new(
            "shaders/mul_mm_template.metallib",
        ))
        .unwrap();

    let seed = 42u64;
    let k = 4096usize;
    let m = 128usize; // Test with 128 to see pattern

    println!("Testing M=128 (should see pattern in first 64 vs remaining)\n");

    let input_f16 = gen_f16_input(k, 1, seed);
    let weights = gen_iq2_xxs(m, k, seed);

    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM as u64,
        );
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
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

    let my_fn = my_lib
        .get_function("kernel_mul_mm_iq2_xxs_f32", Some(cv.clone()))
        .unwrap();
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_iq2_xxs_f32", Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * 66) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_self = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (k / 256 * 66) as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        r2: 1,
        r3: 1,
    };

    // nr0=4, nsg=4 → rows_per_tg = 16
    let nsg = 4;
    let nr0 = 4;
    let rows_per_tg = nsg * nr0; // 16 rows per threadgroup
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((m + rows_per_tg - 1) / rows_per_tg) as u64, 1, 1);

    println!("Dispatch config:");
    println!("  rows_per_tg = {} (nsg={} * nr0={})", rows_per_tg, nsg, nr0);
    println!("  tg_count = {} threadgroups", tg_count.width);
    println!(
        "  Total rows = {} (should cover {} rows)",
        tg_count.width * rows_per_tg as u64,
        m
    );

    // Run both kernels
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

    // Analyze NaN pattern
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m) };

    println!("\nNaN positions:");
    for i in 0..m {
        let is_nan_self = slice_self[i].is_nan();
        let is_nan_llama = slice_llama[i].is_nan();
        if is_nan_self || is_nan_llama {
            println!(
                "  [{}] self={:.6} (NaN: {}), llama={:.6} (NaN: {})",
                i, slice_self[i], is_nan_self, slice_llama[i], is_nan_llama
            );
        }
    }

    // Group analysis
    println!("\nNaN count by groups of 16 rows (one threadgroup):");
    for tg in 0..(m / rows_per_tg + 1) {
        let start = tg * rows_per_tg;
        let end = ((tg + 1) * rows_per_tg).min(m);
        if start >= m {
            break;
        }

        let nan_self =
            slice_self[start..end].iter().filter(|&&x| x.is_nan()).count();
        let nan_llama =
            slice_llama[start..end].iter().filter(|&&x| x.is_nan()).count();
        println!(
            "  TG[{}] rows [{}..{}]: self={} NaN, llama={} NaN",
            tg, start, end, nan_self, nan_llama
        );
    }

    println!(
        "\nTotal NaN: self={}, llama={}",
        slice_self.iter().filter(|&&x| x.is_nan()).count(),
        slice_llama.iter().filter(|&&x| x.is_nan()).count()
    );
}
