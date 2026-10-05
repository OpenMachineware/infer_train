// GEMM benchmark for all quantization formats with N>1 (true GEMM)
// Tests mul_mm_standalone.metal vs llama.cpp mul_mm_template.metallib

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_diff, print_stats};

const QK_K: usize = 256;

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

fn gen_q4_0(m: usize, k: usize, seed: u64) -> Vec<BlockQ4_0> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ4_0 { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_q8_0(m: usize, k: usize, seed: u64) -> Vec<BlockQ8_0> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0i8; 32];
            for j in 0..32 {
                qs[j] = rng.r#gen::<i8>();
            }
            BlockQ8_0 { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_mxfp4(m: usize, k: usize, seed: u64) -> Vec<BlockMXFP4> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockMXFP4 { e: 140, qs } // E8M0 scale = 1.0
        })
        .collect()
}

fn gen_tq2_0(m: usize, k: usize, seed: u64) -> Vec<BlockTQ2_0> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockTQ2_0 { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_tq1_0(m: usize, k: usize, seed: u64) -> Vec<BlockTQ1_0> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 48];
            let mut qh = [0u8; 4];
            for j in 0..48 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..4 {
                qh[j] = rng.r#gen::<u8>();
            }
            BlockTQ1_0 { d: f16::from_f32(1.0).to_bits(), qs, qh }
        })
        .collect()
}

// Parameter structure for llama.cpp mul_mm_template.metal
#[repr(C)]
struct GemmArgs {
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

fn bench_gemm(
    name: &str,
    my_kernel: &str,
    llama_kernel: &str,
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    args: &GemmArgs,
    m: usize,
    n: usize,
    rounds: usize,
) {
    let queue = device.new_command_queue();

    // Function constants for mul_mm (FC = 700)
    const FC_MUL_MM: usize = 700;
    let cv = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
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

    let my_fn = my_lib.get_function(my_kernel, Some(cv.clone())).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    let llama_fn = llama_lib.get_function(llama_kernel, Some(cv)).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    // Create buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_a = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_b = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Threadgroup dispatch for GEMM
    // NR0 = 64, NR1 = 32 per threadgroup
    let tg_size = MTLSize::new(128, 1, 1); // 128 threads per threadgroup (4 simdgroups × 32 threads)
    let tg_count =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);

    // Warmup
    for _ in 0..5 {
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&my_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgs>() as u64,
                args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_a), 0);
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
                std::mem::size_of::<GemmArgs>() as u64,
                args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_b), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    // Verify correctness
    {
        let ptr_a = output_a.contents() as *const f32;
        let ptr_b = output_b.contents() as *const f32;
        let slice_a = unsafe { std::slice::from_raw_parts(ptr_a, m * n) };
        let slice_b = unsafe { std::slice::from_raw_parts(ptr_b, m * n) };

        let non_zero_a: usize = slice_a.iter().filter(|&&x| x != 0.0).count();
        let non_zero_b: usize = slice_b.iter().filter(|&&x| x != 0.0).count();

        let max_diff = slice_a
            .iter()
            .zip(slice_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let sum_a: f64 = slice_a.iter().map(|&x| x as f64).sum();
        let sum_b: f64 = slice_b.iter().map(|&x| x as f64).sum();

        println!(
            "  Verification: non_zero(self)={}, non_zero(llama)={}, max_diff={:.2}, sum_self={:.2}, sum_llama={:.2}",
            non_zero_a, non_zero_b, max_diff, sum_a, sum_b
        );

        if non_zero_a == 0 || non_zero_b == 0 {
            println!("  WARNING: One of the kernels produced all zeros!");
        }
    }

    // Alternating benchmark: A-B-A-B pattern
    let mut my_total = 0u64;
    let mut llama_total = 0u64;
    let mut my_wins = 0usize;

    for i in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_a), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_a = start.elapsed().as_micros() as u64;
        my_total += us_a;

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_b), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_b = start.elapsed().as_micros() as u64;
        llama_total += us_b;

        if us_a < us_b {
            my_wins += 1;
        }

        if (i + 1) % 10 == 0 {
            print!(".");
        }
    }
    println!();

    let my_avg = my_total as f64 / rounds as f64;
    let llama_avg = llama_total as f64 / rounds as f64;
    let ratio = my_avg / llama_avg * 100.0;

    println!("  {}:", name);
    println!("    Self:    {:.2} µs", my_avg);
    println!("    llama.cpp: {:.2} µs", llama_avg);
    println!("    Ratio:   {:.1}% ({} / {} wins)", ratio, my_wins, rounds);
}

fn main() {
    println!("=== GEMM All Formats Benchmark (N>1) ===");
    println!(
        "Comparing mul_mm_standalone.metal vs llama.cpp mul_mm_template.metallib"
    );

    let device = Device::system_default().expect("No Metal device");

    let m = 4096usize;
    let k = 4096usize;
    let n = 128usize; // True GEMM: N > 1
    let rounds = 100usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}, {} rounds\n", m, k, n, rounds);

    // GPU status
    let gpu_before = get_gpu_stats();
    if let Some(ref g) = gpu_before {
        print_stats("=== GPU Status ===", g);
    }

    // Load pre-compiled libraries
    let my_lib_path =
        std::path::Path::new("shaders/mul_mm_standalone.metallib");
    let my_lib = if my_lib_path.exists() {
        device
            .new_library_with_file(my_lib_path)
            .expect("Failed to load mul_mm_standalone.metallib")
    } else {
        println!(
            "WARNING: mul_mm_standalone.metallib not found, skipping self-written tests"
        );
        return;
    };

    let llama_lib_path =
        std::path::Path::new("shaders/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp template library");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    println!("\n=== Standard Q Formats ===");

    // Helper to create args
    let make_args = |k: usize,
                     m: usize,
                     n: usize,
                     block_size: usize,
                     block_bytes: usize|
     -> GemmArgs {
        GemmArgs {
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
        }
    };

    // Q4_0
    println!("\n--- Q4_0 ---");
    let weights = gen_q4_0(m, k, seed);
    let args = make_args(k, m, n, 32, 18);
    bench_gemm(
        "Q4_0",
        "kernel_mul_mm_q4_0_f32",
        "kernel_mul_mm_q4_0_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        rounds,
    );

    // Q8_0
    println!("\n--- Q8_0 ---");
    let weights = gen_q8_0(m, k, seed);
    let args = make_args(k, m, n, 32, 34);
    bench_gemm(
        "Q8_0",
        "kernel_mul_mm_q8_0_f32",
        "kernel_mul_mm_q8_0_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        rounds,
    );

    println!("\n=== Special Formats ===");

    // MXFP4
    println!("\n--- MXFP4 ---");
    let weights = gen_mxfp4(m, k, seed);
    let args = make_args(k, m, n, 32, 17);
    bench_gemm(
        "MXFP4",
        "kernel_mul_mm_mxfp4_f32",
        "kernel_mul_mm_mxfp4_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        rounds,
    );

    println!("\n=== TQ Formats (GEMM) ===");

    // TQ2_0
    println!("\n--- TQ2_0 ---");
    let weights = gen_tq2_0(m, k, seed);
    let args = make_args(k, m, n, 256, 66);
    bench_gemm(
        "TQ2_0",
        "kernel_mul_mm_tq2_0_f32",
        "kernel_mul_mm_tq2_0_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        rounds,
    );

    // TQ1_0 (llama.cpp doesn't have TQ1_0 GEMM)
    println!("\n--- TQ1_0 ---");
    println!(
        "  Note: llama.cpp does not have TQ1_0 GEMM kernel, only self-written"
    );
    let weights = gen_tq1_0(m, k, seed);
    let args = make_args(k, m, n, 256, 54);
    // Only test self-written kernel
    let my_fn = my_lib.get_function("kernel_mul_mm_tq1_0_f32", None).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let queue = device.new_command_queue();
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
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
    let tg_size = MTLSize::new(128, 4, 1);
    let tg_count =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);

    let mut total = 0u64;
    for i in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
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
        total += start.elapsed().as_micros() as u64;
        if (i + 1) % 10 == 0 {
            print!(".");
        }
    }
    println!();
    println!(
        "  TQ1_0 (self-written only): {:.2} µs avg",
        total as f64 / rounds as f64
    );

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
