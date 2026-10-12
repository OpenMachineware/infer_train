// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Precise GEMM alternating benchmark with GPU frequency normalization
// Compares self-written kernels against llama.cpp kernels
// No excuses: if ratio < 100%, kernel is slower

use half::f16;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_diff, print_stats};

const QK: usize = 32;
const QK_K: usize = 256;

fn expand_includes(
    source: &str,
    base_dir: &std::path::Path,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();
    let mut skip_embed_block = false;
    let mut skip_endif_count = 0;
    let mut after_embed_else = false;

    for line in source.lines() {
        if line.contains("GGML_METAL_EMBED_LIBRARY") {
            skip_embed_block = true;
            skip_endif_count = 1;
            result.push_str("// Skipping embed block\n");
            continue;
        }
        if skip_embed_block {
            if line.starts_with("#if") {
                skip_endif_count += 1;
            } else if line.starts_with("#else") {
                skip_embed_block = false;
                after_embed_else = true;
                result.push_str("// End of embed block\n");
            } else if line.starts_with("#endif") {
                skip_endif_count -= 1;
                if skip_endif_count == 0 {
                    skip_embed_block = false;
                }
            }
            continue;
        }
        if after_embed_else && line.starts_with("#endif") {
            after_embed_else = false;
            continue;
        }

        if line.starts_with("#include \"") {
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];
            if !visited.contains(&filename.to_string()) {
                visited.push(filename.to_string());
                let search_paths = vec![
                    base_dir.join(filename),
                    base_dir.parent().unwrap().join(filename),
                    base_dir.parent().unwrap().parent().unwrap().join(filename),
                ];
                for include_path in &search_paths {
                    if let Ok(content) = std::fs::read_to_string(include_path) {
                        let new_base = include_path.parent().unwrap();
                        result.push_str(&expand_includes(
                            &content.replace("#pragma once", ""),
                            new_base,
                            visited,
                        ));
                        break;
                    }
                }
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

#[repr(C)]
struct GemmArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
}

fn create_function_constants() -> metal::FunctionConstantValues {
    let cv = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
    unsafe {
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            0,
        );
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            1,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            2,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            3,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            4,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            5,
        );
    }
    cv
}

fn bench_format(
    name: &str,
    kernel_name: &str,
    lib: &metal::Library,
    weights_bytes: &[u8],
    input_f16: &[u16],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
    rounds: usize,
) -> (f64, f64, f64, usize) {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let cv = create_function_constants();
    let kernel = lib.get_function(kernel_name, Some(cv)).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights_bytes.as_ptr() as *const c_void,
        weights_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let nb = if block_size >= 84 { k / QK_K } else { k / QK };

    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        ne03: 1,
        nb00: block_size as u64,
        nb01: (nb * block_size) as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        ne13: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        ne2: 1,
        ne3: 1,
        nb0: 4,
        nb1: (m * 4) as u64,
        nb2: 0,
        nb3: 0,
    };

    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((m + 127) / 128) as u64, ((n + 31) / 32) as u64, 1);

    // Warmup
    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark with GPU frequency normalization
    let mut total_us = 0u64;
    let mut freq_sum = 0.0;

    for i in 0..rounds {
        // Get GPU freq before
        let gpu_before = get_gpu_stats();
        let freq_before =
            gpu_before.map(|g| g.gpu_freq_mhz as f64).unwrap_or(1.0);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();

        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us = start.elapsed().as_micros() as u64;

        // Get GPU freq after
        let gpu_after = get_gpu_stats();
        let freq_after =
            gpu_after.map(|g| g.gpu_freq_mhz as f64).unwrap_or(1.0);

        total_us += us;
        freq_sum += (freq_before + freq_after) / 2.0;

        if (i + 1) % 25 == 0 {
            print!(".");
        }
    }

    let avg_us = total_us as f64 / rounds as f64;
    let avg_freq = freq_sum / rounds as f64;
    let normalized = avg_us / avg_freq;

    (avg_us, normalized, avg_freq, rounds)
}

fn main() {
    println!("=== GEMM Precise Alternating Benchmark ===");
    println!("Using GPU frequency normalization - NO EXCUSES\n");

    let device = Device::system_default().expect("No Metal device");

    let m = 4096usize;
    let k = 4096usize;
    let n = 128usize;
    let rounds = 200usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}, {} rounds per format\n", m, k, n, rounds);

    // GPU status before
    let gpu_before = get_gpu_stats();
    if let Some(ref g) = gpu_before {
        print_stats("=== GPU Status Before ===", g);
    }

    // Load llama.cpp library (expand includes)
    let llama_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("llama.cpp-0.6.0/ggml/src/ggml-metal/kernels");
    let llama_raw =
        std::fs::read_to_string(llama_dir.join("mul_mv.metal")).unwrap();
    let llama_source = expand_includes(
        &llama_raw,
        &llama_dir,
        &mut vec!["mul_mv.metal".to_string()],
    );
    let compile_opts = metal::CompileOptions::new();
    let llama_lib = device
        .new_library_with_source(&llama_source, &compile_opts)
        .expect("Failed to compile llama.cpp library");

    // Load self-written library (inline blocks.h with Metal define)
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let ggml_common =
        std::fs::read_to_string(shader_dir.join("blocks.h")).unwrap();
    let iq_grid_tables =
        std::fs::read_to_string(shader_dir.join("iq_grid_tables.h")).unwrap();
    let self_base =
        std::fs::read_to_string(shader_dir.join("gemm.metal")).unwrap();

    let iq_extra_tables = r#"
constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
constexpr constant static int8_t ksigns_iq2xs[128] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,
     1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,
     1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,
     1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
};
"#;

    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let self_source = format!(
        "{}\n{}\n{}\n{}",
        ggml_common_metal, iq_extra_tables, iq_grid_tables, self_base
    );
    let self_lib = device
        .new_library_with_source(&self_source, &compile_opts)
        .expect("Failed to compile self-written library");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    // Formats to test: (name, llama_kernel, self_kernel, block_size, nr0)
    // llama.cpp uses kernel_mul_mv_*_f32
    // self-written uses kernel_mul_mm_*_f32
    let formats: Vec<(&str, &str, &str, usize, usize)> = vec![
        ("Q2_K", "kernel_mul_mv_q2_K_f32", "kernel_mul_mm_q2_K_f32", 84, 4),
        ("Q3_K", "kernel_mul_mv_q3_K_f32", "kernel_mul_mm_q3_K_f32", 110, 2),
        ("Q4_K", "kernel_mul_mv_q4_K_f32", "kernel_mul_mm_q4_K_f32", 144, 2),
        ("Q5_K", "kernel_mul_mv_q5_K_f32", "kernel_mul_mm_q5_K_f32", 176, 1),
        ("Q6_K", "kernel_mul_mv_q6_K_f32", "kernel_mul_mm_q6_K_f32", 210, 2),
    ];

    println!(
        "\n{:<10} | {:>12} | {:>12} | {:>6} | {:>8} | Status",
        "Format", "Self us/MHz", "Llama us/MHz", "Ratio", "Freq MHz"
    );
    println!("{}", "-".repeat(65));

    for (name, llama_kernel, self_kernel, block_size, _nr0) in &formats {
        // Generate weights with valid data
        let nb = k / QK_K;
        let weights_size = m * nb * block_size;
        let mut rng = StdRng::seed_from_u64(seed);
        let weights: Vec<u8> = (0..weights_size)
            .map(|i| {
                let byte_in_block = i % block_size;
                // Ensure valid scales and d values
                match *name {
                    "Q2_K" => {
                        // Q2_K: scales[16], qs[64], d[2], dmin[2]
                        if byte_in_block < 16 {
                            (rng.r#gen::<u8>() % 63 + 1)
                        }
                        // scales
                        else if byte_in_block == 80 || byte_in_block == 82 {
                            0x3C
                        }
                        // d, dmin
                        else {
                            rng.r#gen::<u8>()
                        }
                    }
                    "Q3_K" => {
                        // Q3_K: hmask[32], qs[64], scales[12], d[2]
                        if byte_in_block >= 96 && byte_in_block < 108 {
                            (rng.r#gen::<u8>() % 63 + 1)
                        } else if byte_in_block == 108 {
                            0x3C
                        } else {
                            rng.r#gen::<u8>()
                        }
                    }
                    "Q4_K" => {
                        // Q4_K: d[2], dmin[2], scales[12], qs[128]
                        if byte_in_block < 2 {
                            0x3C
                        }
                        // d
                        else if byte_in_block >= 2 && byte_in_block < 4 {
                            0x0C
                        }
                        // dmin
                        else if byte_in_block >= 4 && byte_in_block < 16 {
                            (rng.r#gen::<u8>() % 63 + 1)
                        }
                        // scales
                        else {
                            rng.r#gen::<u8>()
                        }
                    }
                    "Q5_K" => {
                        // Q5_K: d[2], dmin[2], scales[12], qh[32], qs[128]
                        if byte_in_block < 2 {
                            0x3C
                        } else if byte_in_block >= 2 && byte_in_block < 4 {
                            0x0C
                        } else if byte_in_block >= 4 && byte_in_block < 16 {
                            (rng.r#gen::<u8>() % 63 + 1)
                        } else {
                            rng.r#gen::<u8>()
                        }
                    }
                    "Q6_K" => {
                        // Q6_K: ql[128], qh[64], scales[16], d[2]
                        if byte_in_block >= 192 && byte_in_block < 208 {
                            (rng.r#gen::<u8>() % 31 + 1)
                        } else if byte_in_block == 208 {
                            0x3C
                        } else {
                            rng.r#gen::<u8>()
                        }
                    }
                    _ => rng.r#gen::<u8>(),
                }
            })
            .collect();

        // Bench self-written
        let (_, self_norm, self_freq, _) = bench_format(
            name,
            self_kernel,
            &self_lib,
            &weights,
            &input_f16,
            m,
            n,
            k,
            *block_size,
            rounds,
        );

        // Bench llama.cpp
        let (_, llama_norm, llama_freq, _) = bench_format(
            name,
            llama_kernel,
            &llama_lib,
            &weights,
            &input_f16,
            m,
            n,
            k,
            *block_size,
            rounds,
        );

        let ratio = self_norm / llama_norm * 100.0;
        let status = if ratio < 99.0 {
            format!("❌ SLOWER")
        } else if ratio > 101.0 {
            format!("✓ FASTER")
        } else {
            "TIED".to_string()
        };

        println!(
            "{:<10} | {:>12.3} | {:>12.3} | {:>5.1}% | {:>7.0} | {}",
            name,
            self_norm,
            llama_norm,
            ratio,
            (self_freq + llama_freq) / 2.0,
            status
        );
    }

    // GPU status after
    let gpu_after = get_gpu_stats();
    if let (Some(before), Some(after)) =
        (gpu_before.as_ref(), gpu_after.as_ref())
    {
        print_diff(&before.diff(after));
    }

    println!("\nDone. Ratio < 99% means kernel is slower - no excuses.");
}
