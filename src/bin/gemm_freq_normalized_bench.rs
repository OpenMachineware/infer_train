// GEMM performance benchmark with GPU frequency normalization
// Tests all formats and normalizes by GPU frequency for fair comparison

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_diff, print_stats};

const QK: usize = 32;
const QK_K: usize = 256;

fn gen_f32_matrix(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

// GEMM args
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

fn bench_gemm_format(
    name: &str,
    kernel_name: &str,
    lib: &metal::Library,
    weights_buffer: &metal::Buffer,
    input_buffer: &metal::Buffer,
    output_buffer: &metal::Buffer,
    args: &GemmArgs,
    m: usize,
    n: usize,
    rounds: usize,
) -> (f64, f64) {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

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

    let kernel = lib.get_function(kernel_name, Some(cv)).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    // Threadgroup size
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
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(weights_buffer), 0);
        enc.set_buffer(2, Some(input_buffer), 0);
        enc.set_buffer(3, Some(output_buffer), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    let mut total_us = 0u64;
    for i in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(weights_buffer), 0);
        enc.set_buffer(2, Some(input_buffer), 0);
        enc.set_buffer(3, Some(output_buffer), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();

        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us = start.elapsed().as_micros() as u64;
        total_us += us;

        if (i + 1) % 20 == 0 {
            print!(".");
        }
    }
    println!(" {:.0} µs", total_us / rounds as u64);

    let avg_us = total_us as f64 / rounds as f64;

    // Get GPU frequency for normalization
    let gpu = get_gpu_stats();
    let norm = if let Some(ref g) = gpu {
        avg_us / g.gpu_freq_mhz as f64
    } else {
        avg_us
    };

    (avg_us, norm)
}

fn main() {
    println!("=== GEMM Performance Benchmark (Frequency Normalized) ===");

    let device = Device::system_default().expect("No Metal device");

    let m = 4096usize;
    let k = 4096usize;
    let n = 128usize;
    let rounds = 100usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}, {} rounds\n", m, k, n, rounds);

    // GPU status before
    let gpu_before = get_gpu_stats();
    if let Some(ref g) = gpu_before {
        print_stats("=== GPU Status Before Benchmark ===", g);
    }

    // Load libraries
    let llama_lib_source =
        include_str!("../../shaders/mul_mm_standalone.metal");
    let compile_opts = metal::CompileOptions::new();
    let llama_lib = device
        .new_library_with_source(llama_lib_source, &compile_opts)
        .expect("Failed to compile GEMM library");

    // Generate test data
    let input_f16 = gen_f16_input(k, n, seed);
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Test formats
    println!("\n=== K-quant GEMM ===");
    let k_quant_formats: [(&str, &str, usize); 5] = [
        ("Q2_K", "kernel_mul_mm_q2_K_f32", 84),
        ("Q3_K", "kernel_mul_mm_q3_K_f32", 110),
        ("Q4_K", "kernel_mul_mm_q4_K_f32", 144),
        ("Q5_K", "kernel_mul_mm_q5_K_f32", 176),
        ("Q6_K", "kernel_mul_mm_q6_K_f32", 210),
    ];

    for &(name, kernel, block_size) in k_quant_formats.iter() {
        println!("\n{}:", name);

        // Generate weights
        let nb = k / QK_K;
        let mut weights = Vec::with_capacity(m * nb * block_size);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..m * nb * block_size {
            weights.push(rng.r#gen::<u8>());
        }
        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            weights.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

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

        let (avg_us, norm) = bench_gemm_format(
            name,
            kernel,
            &llama_lib,
            &weights_buffer,
            &input_buffer,
            &output_buffer,
            &args,
            m,
            n,
            rounds,
        );
        println!("  Avg: {:.0} µs, Normalized: {:.3} µs/MHz", avg_us, norm);
    }

    println!("\n=== Standard Q GEMM ===");
    let std_formats: [(&str, &str, usize); 5] = [
        ("Q4_0", "kernel_mul_mm_q4_0_f32", 18),
        ("Q8_0", "kernel_mul_mm_q8_0_f32", 34),
        ("Q4_1", "kernel_mul_mm_q4_1_f32", 20),
        ("Q5_0", "kernel_mul_mm_q5_0_f32", 22),
        ("Q5_1", "kernel_mul_mm_q5_1_f32", 24),
    ];

    for &(name, kernel, block_size) in std_formats.iter() {
        println!("\n{}:", name);

        let nb = k / QK;
        let mut weights = Vec::with_capacity(m * nb * block_size);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..m * nb * block_size {
            weights.push(rng.r#gen::<u8>());
        }
        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            weights.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

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

        let (avg_us, norm) = bench_gemm_format(
            name,
            kernel,
            &llama_lib,
            &weights_buffer,
            &input_buffer,
            &output_buffer,
            &args,
            m,
            n,
            rounds,
        );
        println!("  Avg: {:.0} µs, Normalized: {:.3} µs/MHz", avg_us, norm);
    }

    println!("\n=== MXFP4 and IQ4_NL GEMM ===");
    let special_formats: [(&str, &str, usize); 2] = [
        ("MXFP4", "kernel_mul_mm_mxfp4_f32", 17),
        ("IQ4_NL", "kernel_mul_mm_iq4_nl_f32", 18),
    ];

    for &(name, kernel, block_size) in special_formats.iter() {
        println!("\n{}:", name);

        let nb = k / QK;
        let mut weights = Vec::with_capacity(m * nb * block_size);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..m * nb * block_size {
            weights.push(rng.r#gen::<u8>());
        }
        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            weights.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

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

        let (avg_us, norm) = bench_gemm_format(
            name,
            kernel,
            &llama_lib,
            &weights_buffer,
            &input_buffer,
            &output_buffer,
            &args,
            m,
            n,
            rounds,
        );
        println!("  Avg: {:.0} µs, Normalized: {:.3} µs/MHz", avg_us, norm);
    }

    // GPU status after
    let gpu_after = get_gpu_stats();
    if let (Some(before), Some(after)) =
        (gpu_before.as_ref(), gpu_after.as_ref())
    {
        let diff = before.diff(after);
        print_diff(&diff);
    }

    println!("\nDone.");
}
