// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare all K-quant MV kernels vs llama.cpp
use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ2K {
    scales: [u8; 32],
    qs: [u8; 128],
    d: u16,
    dmin: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
    d: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ5K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
    qh: [u8; 64],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    d: u16,
}

#[repr(C)]
struct MvArgs {
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

fn create_cv() -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let nsg: i16 = 2;
        let ne12: i16 = 1;
        let r2: i16 = 1;
        let r3: i16 = 1;
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn expand_includes(
    source: &str,
    shader_dir: &std::path::PathBuf,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();
    for line in source.lines() {
        if line.starts_with("#include \"") {
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];
            if visited.contains(&filename.to_string()) {
                continue;
            }
            visited.push(filename.to_string());
            let include_path = shader_dir.join(filename);
            if let Ok(content) = std::fs::read_to_string(&include_path) {
                result.push_str(&expand_includes(
                    &content.replace("#pragma once", "// #pragma once"),
                    shader_dir,
                    visited,
                ));
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Load llama.cpp kernel library
    let llama_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("llama.cpp-0.6.0/build/bin/kernels");
    let llama_source =
        std::fs::read_to_string(llama_dir.join("mul_mv.metal")).unwrap();
    let llama_lib = device
        .new_library_with_source(&llama_source, &compile_options)
        .unwrap();

    // Load self-written template
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source =
        std::fs::read_to_string(shader_dir.join("gemm.metal")).unwrap();
    let my_lib =
        device.new_library_with_source(&my_source, &compile_options).unwrap();

    let m = 4096usize;
    let k = 4096usize;
    let iterations = 5000;

    let input: Vec<f32> = (0..k).map(|i| (i % 10) as f32 * 0.1).collect();
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cv = create_cv();
    let output_size = m * 4;
    let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
    let tg_count = MTLSize { width: (m as u64 + 3) / 4, height: 1, depth: 1 };

    println!(
        "=== K-quant MV Performance vs llama.cpp (M={}, K={}, {} iter) ===\n",
        m, k, iterations
    );
    println!(
        "{:8} | {:>10} | {:>10} | {:>6} | {:>8}",
        "Format", "llama.cpp", "self", "Ratio", "Correct"
    );

    // Test each format
    let formats: [(&str, usize, usize); 5] = [
        ("Q2_K", 164, 256),
        ("Q3_K", 110, 256),
        ("Q4_K", 144, 256),
        ("Q5_K", 208, 256),
        ("Q6_K", 212, 256),
    ];

    for (name, block_size, qk) in formats {
        let kernel_name = format!("kernel_mul_mm_{}_f32", name.to_lowercase());
        let nb = k / qk;
        let nb01 = (block_size * nb) as u64;
        let weights_size = m * nb * block_size;
        let weights_buffer = device.new_buffer(
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: block_size as u64,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let llama_kernel =
            match llama_lib.get_function(&kernel_name, Some(cv.clone())) {
                Ok(k) => k,
                Err(_) => {
                    println!("{:8} | NOT FOUND in llama.cpp", name);
                    continue;
                }
            };
        let my_kernel =
            match my_lib.get_function(&kernel_name, Some(cv.clone())) {
                Ok(k) => k,
                Err(_) => {
                    println!("{:8} | NOT FOUND in self", name);
                    continue;
                }
            };

        let llama_pipeline = device
            .new_compute_pipeline_state_with_function(&llama_kernel)
            .unwrap();
        let my_pipeline = device
            .new_compute_pipeline_state_with_function(&my_kernel)
            .unwrap();

        // Warmup
        for _ in 0..100 {
            for pipeline in [&llama_pipeline, &my_pipeline] {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                let out = device.new_buffer(
                    output_size as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                enc.set_compute_pipeline_state(pipeline);
                enc.set_bytes(
                    0,
                    std::mem::size_of::<MvArgs>() as u64,
                    &args as *const MvArgs as *const c_void,
                );
                enc.set_buffer(1, Some(&weights_buffer), 0);
                enc.set_buffer(2, Some(&input_buffer), 0);
                enc.set_buffer(3, Some(&out), 0);
                enc.dispatch_thread_groups(tg_count, tg_size);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }

        // Benchmark llama.cpp
        let mut llama_time = 0u128;
        let llama_output = device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        for _ in 0..iterations {
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&llama_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&llama_output), 0);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            llama_time += start.elapsed().as_micros();
        }

        // Benchmark self-written
        let mut my_time = 0u128;
        let my_output = device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        for _ in 0..iterations {
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&my_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&my_output), 0);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            my_time += start.elapsed().as_micros();
        }

        // Check correctness
        let llama_ptr = llama_output.contents() as *const f32;
        let my_ptr = my_output.contents() as *const f32;
        let correct: usize = (0..m)
            .filter(|&i| {
                let lv = unsafe { *llama_ptr.add(i) };
                let mv = unsafe { *my_ptr.add(i) };
                if lv.abs() > 1e-6 {
                    (lv - mv).abs() / lv.abs() < 0.01
                } else {
                    (lv - mv).abs() < 1e-6
                }
            })
            .count();

        let llama_avg = llama_time as f64 / iterations as f64;
        let my_avg = my_time as f64 / iterations as f64;
        let ratio = llama_avg / my_avg * 100.0;
        let correct_pct = correct * 100 / m;

        println!(
            "{:8} | {:>8.2} us | {:>8.2} us | {:>5.1}% | {:>6}% ({}/{})",
            name, llama_avg, my_avg, ratio, correct_pct, correct, m
        );
    }
}
