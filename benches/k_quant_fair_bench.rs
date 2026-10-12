// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Fair alternating benchmark: self-written vs llama.cpp
// Each run alternates between both kernels to eliminate thermal variance
use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

// Rust block structures for data generation
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
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
    d: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ2K {
    scales: [u8; 16],
    qs: [u8; 64],
    d: u16,
    dmin: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ5K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qh: [u8; 32],
    qs: [u8; 128],
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

// Metal block type definitions for llama.cpp kernel
const METAL_BLOCK_TYPES: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12

typedef half ggml_half;
typedef half2 ggml_half2;

#define QK4_0 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

#define QK8_0 32
typedef struct {
    ggml_half d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    union {
        struct {
            ggml_half d;
            ggml_half dmin;
        };
        ggml_half2 dm;
    };
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    ggml_half d;
} block_q3_K;

typedef struct {
    union {
        struct {
            ggml_half d;
            ggml_half dmin;
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    union {
        struct {
            ggml_half d;
            ggml_half dmin;
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    ggml_half d;
} block_q6_K;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH 32

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};
"#;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Load llama.cpp kernel source
    let llama_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("llama.cpp-0.6.0/ggml/src/ggml-metal/kernels");
    let llama_raw =
        std::fs::read_to_string(llama_dir.join("mul_mv.metal")).unwrap();

    // Prepend block types to llama.cpp source
    let llama_source = format!("{}\n{}", METAL_BLOCK_TYPES, llama_raw);
    let llama_lib = device
        .new_library_with_source(&llama_source, &compile_options)
        .expect("Failed to compile llama.cpp kernel");

    // Load self-written kernel
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source =
        std::fs::read_to_string(shader_dir.join("gemm.metal")).unwrap();
    let my_lib = device
        .new_library_with_source(&my_source, &compile_options)
        .expect("Failed to compile self-written kernel");

    let m = 1024usize;
    let k = 4096usize;
    let rounds = 100;

    // Test all K-quant formats
    let formats: Vec<(&str, usize, &str, &str)> = vec![
        ("Q2_K", 84, "kernel_mul_mv_q2_K_f32", "kernel_mul_mm_q2_K_f32"),
        ("Q3_K", 110, "kernel_mul_mv_q3_K_f32", "kernel_mul_mm_q3_K_f32"),
        ("Q4_K", 144, "kernel_mul_mv_q4_K_f32", "kernel_mul_mm_q4_K_f32"),
        ("Q5_K", 176, "kernel_mul_mv_q5_K_f32", "kernel_mul_mm_q5_K_f32"),
        ("Q6_K", 210, "kernel_mul_mv_q6_K_f32", "kernel_mul_mm_q6_K_f32"),
    ];

    println!("=== K-quant Fair Alternating Benchmark ===");
    println!("M={}, K={}, {} alternating rounds\n", m, k, rounds);

    for (name, block_size, llama_kernel, my_kernel) in &formats {
        let nb = k / 256;

        // Generate test data
        let weights = generate_weights(m, nb, *block_size);
        let input: Vec<f32> = (0..k).map(|i| (i % 100) as f32 * 0.01).collect();

        // Create buffers
        let weights_buf = device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            (m * nb * block_size) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let input_buf = device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (k * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let output_llama = device.new_buffer(
            (m * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let output_mine = device.new_buffer(
            (m * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        // Create pipelines
        let llama_pipe = llama_lib
            .get_function(*llama_kernel, Some(create_cv()))
            .and_then(|f| device.new_compute_pipeline_state_with_function(&f))
            .expect("Failed to create llama.cpp pipeline");

        let my_pipe = my_lib
            .get_function(*my_kernel, Some(create_cv()))
            .and_then(|f| device.new_compute_pipeline_state_with_function(&f))
            .expect("Failed to create self-written pipeline");

        // Setup args
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: *block_size as u64,
            nb01: (*block_size * nb) as u64,
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
            nr0: 0,
            r2: 1,
            r3: 1,
        };
        let args_buf = device.new_buffer_with_data(
            &args as *const MvArgs as *const c_void,
            std::mem::size_of::<MvArgs>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        // Alternating benchmark
        let mut llama_times = Vec::new();
        let mut my_times = Vec::new();

        for i in 0..rounds {
            let cmd_buf = queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();

            encoder.set_buffer(0, Some(&args_buf), 0);
            encoder.set_buffer(1, Some(&weights_buf), 0);
            encoder.set_buffer(2, Some(&input_buf), 0);

            if i % 2 == 0 {
                // Run llama.cpp kernel
                encoder.set_buffer(3, Some(&output_llama), 0);
                encoder.set_compute_pipeline_state(&llama_pipe);
                encoder.dispatch_thread_groups(
                    MTLSize { width: 1, height: 1, depth: 1 },
                    MTLSize { width: 32, height: 1, depth: 1 },
                );
                let start = Instant::now();
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
                llama_times.push(start.elapsed().as_nanos() as f64);
            } else {
                // Run self-written kernel
                encoder.set_buffer(3, Some(&output_mine), 0);
                encoder.set_compute_pipeline_state(&my_pipe);
                encoder.dispatch_thread_groups(
                    MTLSize { width: 1, height: 1, depth: 1 },
                    MTLSize { width: 32, height: 1, depth: 1 },
                );
                let start = Instant::now();
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
                my_times.push(start.elapsed().as_nanos() as f64);
            }
        }

        let llama_avg: f64 =
            llama_times.iter().sum::<f64>() / llama_times.len() as f64;
        let my_avg: f64 = my_times.iter().sum::<f64>() / my_times.len() as f64;
        let ratio = my_avg / llama_avg;

        println!(
            "{:6} {:8.2} us  {:8.2} us  {:6.1}% {}",
            name,
            llama_avg / 1000.0,
            my_avg / 1000.0,
            ratio * 100.0,
            if ratio < 1.0 {
                "✓ FASTER"
            } else if ratio < 1.02 {
                "≈ TIED"
            } else {
                "✗ SLOWER"
            }
        );
    }
}

fn generate_weights(m: usize, nb: usize, block_size: usize) -> Vec<u8> {
    let mut data = vec![0u8; m * nb * block_size];
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }
    data
}
