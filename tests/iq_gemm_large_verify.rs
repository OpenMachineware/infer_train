// Verify IQ GEMM correctness at large sizes (M=4096)
// Investigate NaN and large max_diff issues
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

fn gen_iq3_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3XXS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 96];
            for j in 0..96 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ3XXS { d: f16::from_f32(1.0).to_bits(), qs }
        })
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
            BlockIQ2XXS { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_iq2_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut scales = [0u8; 8];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u8>();
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ2S { d: f16::from_f32(1.0).to_bits(), qs, qh, scales }
        })
        .collect()
}

fn gen_iq3_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut signs = [0u8; 32];
            let mut scales = [0u8; 4];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..32 {
                signs[j] = rng.r#gen::<u8>();
            }
            for j in 0..4 {
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ3S { d: f16::from_f32(1.0).to_bits(), qs, qh, signs, scales }
        })
        .collect()
}

fn gen_iq4_xs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4XS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut scales_l = [0u8; 4];
            for j in 0..128 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..4 {
                scales_l[j] = rng.r#gen::<u8>();
            }
            BlockIQ4XS {
                d: f16::from_f32(1.0).to_bits(),
                scales_h: f16::from_f32(1.0).to_bits(),
                scales_l,
                qs,
            }
        })
        .collect()
}

// Parameter structure for llama.cpp mul_mm_template.metal
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

fn test_format<T: bytemuck::Pod + bytemuck::Zeroable>(
    name: &str,
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[T],
    input_f16: &[u16],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    block_bytes: usize,
    nr0: usize,
) {
    println!("\n=== {} (M={}, K={}) ===", name, m, k);

    let queue = device.new_command_queue();

    // Function constants
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

    let kernel_name = format!("kernel_mul_mm_{}_f32", name.to_lowercase());
    let my_fn = my_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();
    let llama_fn = llama_lib.get_function(&kernel_name, Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * block_size) as u64,
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
        nb01: (k / block_size * block_bytes) as u64,
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

    // Dispatch
    let nsg = 4;
    let rows_per_tg = nsg * nr0;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((m + rows_per_tg - 1) / rows_per_tg) as u64, n as u64, 1);

    // Run self kernel
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

    // Run llama kernel
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

    // Compare
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    // Count NaN/Inf
    let nan_self = slice_self.iter().filter(|&&x| x.is_nan()).count();
    let nan_llama = slice_llama.iter().filter(|&&x| x.is_nan()).count();
    let inf_self = slice_self.iter().filter(|&&x| x.is_infinite()).count();
    let inf_llama = slice_llama.iter().filter(|&&x| x.is_infinite()).count();

    // Only compare finite values
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut diff_count = 0;
    for (i, (&a, &b)) in slice_self.iter().zip(slice_llama.iter()).enumerate() {
        if a.is_finite() && b.is_finite() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
            if diff > 1e-5 {
                diff_count += 1;
            }
        }
    }

    let non_zero_self =
        slice_self.iter().filter(|&&x| x != 0.0 && x.is_finite()).count();
    let non_zero_llama =
        slice_llama.iter().filter(|&&x| x != 0.0 && x.is_finite()).count();

    println!("  Output size: {} elements", m * n);
    println!("  NaN: self={}, llama={}", nan_self, nan_llama);
    println!("  Inf: self={}, llama={}", inf_self, inf_llama);
    println!(
        "  Non-zero (finite): self={}, llama={}",
        non_zero_self, non_zero_llama
    );
    println!(
        "  Max diff (finite values): {:.6} at index {}",
        max_diff, max_diff_idx
    );
    println!("  Elements with diff > 1e-5: {}", diff_count);

    if nan_self > 0 || nan_llama > 0 {
        println!(
            "  ⚠️ NaN detected - investigating first few NaN positions..."
        );
        for (i, (&a, &b)) in
            slice_self.iter().zip(slice_llama.iter()).enumerate().take(20)
        {
            if a.is_nan() || b.is_nan() {
                println!("    [{}] self={:.6}, llama={:.6}", i, a, b);
            }
        }
    }

    if max_diff > 1e-5 {
        println!("  ⚠️ Large diff detected at index {}:", max_diff_idx);
        println!(
            "    self[{}] = {:.6}",
            max_diff_idx, slice_self[max_diff_idx]
        );
        println!(
            "    llama[{}] = {:.6}",
            max_diff_idx, slice_llama[max_diff_idx]
        );

        // Show surrounding values
        let start = (max_diff_idx as isize - 2).max(0) as usize;
        let end = (max_diff_idx + 3).min(m * n);
        println!("    Context:");
        for i in start..end {
            println!(
                "      [{}] self={:.6}, llama={:.6}, diff={:.6}",
                i,
                slice_self[i],
                slice_llama[i],
                (slice_self[i] - slice_llama[i]).abs()
            );
        }
    }
}

fn main() {
    println!("=== IQ GEMM Large Size Verification ===");
    println!("Investigating M=4096 correctness issues\n");

    let device = Device::system_default().expect("No Metal device");

    // Load libraries
    let my_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_standalone.metallib");
    let my_lib = device
        .new_library_with_file(my_lib_path)
        .expect("Failed to load self-written GEMM library");

    let llama_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp template library");

    let seed = 42u64;

    // Test different M sizes
    let m_sizes = [64, 256, 1024, 4096];
    let k = 4096usize;
    let n = 1usize;

    for m in &m_sizes {
        println!("\n========================================");
        println!("Testing with M={}", m);
        println!("========================================");

        let input_f16 = gen_f16_input(k, n, seed);

        // Test formats mentioned in summary with issues
        let weights_iq2_xxs = gen_iq2_xxs(*m, k, seed);
        test_format(
            "IQ2_XXS",
            &device,
            &my_lib,
            &llama_lib,
            &weights_iq2_xxs,
            &input_f16,
            *m,
            k,
            n,
            66,
            256,
            4,
        );

        let weights_iq2_s = gen_iq2_s(*m, k, seed);
        test_format(
            "IQ2_S",
            &device,
            &my_lib,
            &llama_lib,
            &weights_iq2_s,
            &input_f16,
            *m,
            k,
            n,
            82,
            256,
            4,
        );

        let weights_iq3_xxs = gen_iq3_xxs(*m, k, seed);
        test_format(
            "IQ3_XXS",
            &device,
            &my_lib,
            &llama_lib,
            &weights_iq3_xxs,
            &input_f16,
            *m,
            k,
            n,
            98,
            256,
            4,
        );

        let weights_iq3_s = gen_iq3_s(*m, k, seed);
        test_format(
            "IQ3_S",
            &device,
            &my_lib,
            &llama_lib,
            &weights_iq3_s,
            &input_f16,
            *m,
            k,
            n,
            110,
            256,
            4,
        );

        let weights_iq4_xs = gen_iq4_xs(*m, k, seed);
        test_format(
            "IQ4_XS",
            &device,
            &my_lib,
            &llama_lib,
            &weights_iq4_xs,
            &input_f16,
            *m,
            k,
            n,
            136,
            256,
            2,
        );
    }

    println!("\n=== Summary ===");
    println!("All formats tested at M=64, 256, 1024, 4096");
    println!("Looking for:");
    println!("  1. NaN values in output");
    println!("  2. Large max_diff at large M sizes");
    println!("  3. Patterns of divergence");
}
