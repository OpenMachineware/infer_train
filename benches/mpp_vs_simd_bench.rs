// Benchmark: MPP tensor API vs traditional SIMD-group matrix multiply
// Compares llama.cpp's mul_mm compiled with/without GGML_METAL_HAS_TENSOR

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_stats};

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
    println!("=== MPP Tensor API vs SIMD-Group Benchmark ===");
    println!(
        "Comparing llama.cpp kernels compiled with/without GGML_METAL_HAS_TENSOR"
    );

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test dimensions
    let m = 1024usize;
    let k = 1024usize;
    let n = 8usize; // GEMM with N>1 to see MPP benefit
    let rounds = 100usize;
    let seed = 42u64;

    println!("M={}, K={}, N={} (GEMM), {} rounds\n", m, k, n, rounds);

    // GPU status
    let gpu_before = get_gpu_stats();
    if let Some(ref g) = gpu_before {
        print_stats("=== GPU Status ===", g);
    }

    // Load both libraries
    let simd_lib = device
        .new_library_with_file("shaders/mul_mm_template.metallib")
        .expect("Failed to load SIMD library (non-MPP)");

    let mpp_lib = device
        .new_library_with_file("shaders/mul_mm_llama_tensor.metallib")
        .expect("Failed to load MPP library");

    // Function constants
    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let create_cv = || {
        let cv = metal::FunctionConstantValues::new();
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
        cv
    };

    let simd_pipe = device
        .new_compute_pipeline_state_with_function(
            &simd_lib
                .get_function("kernel_mul_mm_iq3_xxs_f32", Some(create_cv()))
                .unwrap(),
        )
        .unwrap();

    let mpp_pipe = device
        .new_compute_pipeline_state_with_function(
            &mpp_lib
                .get_function("kernel_mul_mm_iq3_xxs_f32", Some(create_cv()))
                .unwrap(),
        )
        .unwrap();

    // Generate test data
    let weights = gen_iq3_xxs(m, k, seed);
    let input = gen_f16_input(k, n, seed);

    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (k / 256 * 98) as u64,
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

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<BlockIQ3XXS>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (input.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_simd = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_mpp = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tg_size = MTLSize::new(128, 1, 1);
    let tg_count = MTLSize::new((n + 63) as u64 / 64, (m + 63) as u64 / 64, 1);

    // Warmup
    for _ in 0..5 {
        for (pipe, out) in
            [&simd_pipe, &mpp_pipe].iter().zip([&output_simd, &output_mpp])
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgsMm>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(out), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    // Verify correctness
    {
        let ptr_simd = output_simd.contents() as *const f32;
        let ptr_mpp = output_mpp.contents() as *const f32;
        let slice_simd = unsafe { std::slice::from_raw_parts(ptr_simd, m * n) };
        let slice_mpp = unsafe { std::slice::from_raw_parts(ptr_mpp, m * n) };

        let max_diff = slice_simd
            .iter()
            .zip(slice_mpp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max_diff > 1e-4 {
            println!("WARNING: max_diff = {} (results differ)", max_diff);
        } else {
            println!("Correctness verified: max_diff = {}", max_diff);
        }
    }

    // Alternating benchmark
    println!("\nRunning alternating benchmark (A-B-A-B)...");
    let mut simd_total = 0u64;
    let mut mpp_total = 0u64;
    let mut mpp_wins = 0usize;

    for i in 0..rounds {
        // A: SIMD (non-MPP)
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&simd_pipe);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_simd), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_simd = start.elapsed().as_micros() as u64;
        simd_total += us_simd;

        // B: MPP
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&mpp_pipe);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_mpp), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_mpp = start.elapsed().as_micros() as u64;
        mpp_total += us_mpp;

        if us_mpp < us_simd {
            mpp_wins += 1;
        }

        if (i + 1) % 50 == 0 {
            print!(".");
        }
    }
    println!();

    let simd_avg = simd_total as f64 / rounds as f64;
    let mpp_avg = mpp_total as f64 / rounds as f64;
    let ratio = mpp_avg / simd_avg * 100.0;

    // Frequency normalization
    let gpu_now = get_gpu_stats();
    let freq_mhz = gpu_now.map(|g| g.gpu_freq_mhz as f64).unwrap_or(1.0);

    let simd_per_mhz = simd_avg / freq_mhz;
    let mpp_per_mhz = mpp_avg / freq_mhz;
    let ratio_normalized = mpp_per_mhz / simd_per_mhz * 100.0;

    println!("\n=== Results ===");
    println!(
        "  SIMD (non-MPP): {:.2} µs ({:.3} µs/MHz @ {:.0} MHz)",
        simd_avg, simd_per_mhz, freq_mhz
    );
    println!("  MPP (tensor):   {:.2} µs ({:.3} µs/MHz)", mpp_avg, mpp_per_mhz);
    println!("  Ratio:          {:.1}% (MPP/ SIMD)", ratio);
    println!("  Freq-normalized: {:.1}%", ratio_normalized);
    println!("  MPP wins:       {} / {} rounds", mpp_wins, rounds);

    if ratio < 100.0 {
        println!("\n  MPP is {:.1}% FASTER than SIMD", 100.0 - ratio);
    } else {
        println!("\n  MPP is {:.1}% SLOWER than SIMD", ratio - 100.0);
    }
}
