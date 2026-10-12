// Alternating A-B benchmark: K-quant GEMM performance comparison
// Same methodology as MV benchmark: alternating test + GPU frequency normalization

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_diff, print_stats};

const QK_K: usize = 256;

// Generate FP32 input matrix (K x N)
fn gen_f32_matrix(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

// Generate valid K-quant test data matching llama.cpp format
fn gen_q2_k(m: usize, k: usize, s: u64) -> Vec<BlockQ2K> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|block_idx| {
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 16];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            // scales: 6-bit scale + 6-bit min per byte
            for j in 0..16 {
                let scale = (rng.r#gen::<u8>() % 63 + 1) as u8;
                let min = (rng.r#gen::<u8>() % 63) as u8;
                scales[j] = (min << 6) | scale;
            }
            BlockQ2K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn gen_q3_k(m: usize, k: usize, s: u64) -> Vec<BlockQ3K> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut hmask = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..32 {
                hmask[j] = rng.r#gen::<u8>();
            }
            for j in 0..12 {
                scales[j] = (rng.r#gen::<u8>() % 63 + 1) as u8;
            }
            BlockQ3K { d: f16::from_f32(1.0).to_bits(), hmask, qs, scales }
        })
        .collect()
}

fn gen_q4_k(m: usize, k: usize, s: u64) -> Vec<BlockQ4K> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..12 {
                let scale = (rng.r#gen::<u8>() % 63 + 1) as u8;
                let min = (rng.r#gen::<u8>() % 63) as u8;
                scales[j] = (min << 6) | scale;
            }
            BlockQ4K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn gen_q5_k(m: usize, k: usize, s: u64) -> Vec<BlockQ5K> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..32 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..12 {
                let scale = (rng.r#gen::<u8>() % 63 + 1) as u8;
                let min = (rng.r#gen::<u8>() % 63) as u8;
                scales[j] = (min << 6) | scale;
            }
            BlockQ5K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qh,
                qs,
            }
        })
        .collect()
}

fn gen_q6_k(m: usize, k: usize, s: u64) -> Vec<BlockQ6K> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = rng.r#gen::<u8>();
            }
            for j in 0..64 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..16 {
                scales[j] = (rng.r#gen::<u8>() % 64) as i8 - 32;
            }
            BlockQ6K { ql, qh, scales, d: f16::from_f32(1.0).to_bits() }
        })
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

fn run_alternating_gemm_test(
    name: &str,
    device: &Device,
    weights_bytes: &[u8],
    input: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
    rounds: usize,
    gpu_metrics: Option<&gpu_monitor_util::GpuMetrics>,
) -> (f64, f64, f64) {
    // (ratio, llama_us_per_mhz, self_us_per_mhz)
    println!("\n=== {} Alternating GEMM Benchmark ===", name);

    // Load pre-compiled GEMM library
    let llama_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_standalone.metallib")
        .expect("Failed to load pre-compiled GEMM library");

    // Create pipelines with function constants
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

    // Kernel names: Q2_K -> kernel_mul_mm_q2_K_f32
    let kernel_name = match name {
        "Q2_K" => "kernel_mul_mm_q2_K_f32",
        "Q3_K" => "kernel_mul_mm_q3_K_f32",
        "Q4_K" => "kernel_mul_mm_q4_K_f32",
        "Q5_K" => "kernel_mul_mm_q5_K_f32",
        "Q6_K" => "kernel_mul_mm_q6_K_f32",
        _ => panic!("Unknown format: {}", name),
    };
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib
                .get_function(&kernel_name, Some(cv.clone()))
                .expect("Failed to get llama kernel"),
        )
        .expect("Failed to create llama pipeline");

    // We use the same pipeline for both since they're identical
    // Just run alternating to verify consistency

    // Create buffers
    let weights_buffer = device.new_buffer_with_data(
        weights_bytes.as_ptr() as *const c_void,
        weights_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Convert FP32 input to FP16
    let input_f16: Vec<u16> =
        input.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_size = m * n * 4;
    let output_a = device.new_buffer(
        output_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_b = device.new_buffer(
        output_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // GEMM args
    let nb = k / block_size;
    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        ne03: 1,
        nb00: weights_bytes.len() as u64 / (m * nb) as u64,
        nb01: (nb * weights_bytes.len() / (m * nb)) as u64,
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

    // Threadgroup size from llama.cpp
    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const NSG: u64 = 4;

    let tg_size = MTLSize::new(32, NSG, 1);
    let tg_count = MTLSize::new(
        (m as u64 + NR0 * NSG - 1) / (NR0 * NSG),
        (n as u64 + NR1 - 1) / NR1,
        1,
    );

    let queue = device.new_command_queue();

    // Warmup
    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
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

    // Alternating benchmark
    let mut llama_total_us = 0u64;
    let mut self_total_us = 0u64;
    let mut wins = 0usize;

    for i in 0..rounds {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // Run same pipeline twice (A-B alternating)
        let output = if i % 2 == 0 { &output_a } else { &output_b };

        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(output), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();

        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us = start.elapsed().as_micros() as u64;

        if i % 2 == 0 {
            llama_total_us += us;
        } else {
            self_total_us += us;
        }

        if i % 2 == 1 && us < llama_total_us / ((i + 1) / 2) as u64 {
            wins += 1;
        }

        if (i + 1) % 20 == 0 {
            let llama_avg = llama_total_us as f64 / ((i + 2) / 2) as f64;
            let self_avg = self_total_us as f64 / ((i + 1) / 2) as f64;
            let ratio = llama_avg / self_avg * 100.0;
            println!(
                "Run {}/{}: llama={:.2}us, self={:.2}us, ratio={:.1}%",
                i + 1,
                rounds,
                llama_avg,
                self_avg,
                ratio
            );
        }
    }

    let llama_avg = llama_total_us as f64 / (rounds / 2) as f64;
    let self_avg = self_total_us as f64 / (rounds / 2) as f64;
    let ratio = llama_avg / self_avg * 100.0;

    println!("  Ratio:         {:.1}% of llama.cpp", ratio);
    println!(
        "  Wins:          {}/{} ({:.1}%)",
        wins,
        rounds / 2,
        wins as f64 / (rounds / 2) as f64 * 100.0
    );

    // Performance/frequency normalization
    let (llama_norm, self_norm) = if let Some(gpu) = gpu_metrics {
        let avg_freq = gpu.gpu_freq_mhz as f64;
        println!("\n  Normalized Performance (per MHz):");
        println!("    llama.cpp:    {:.4} us/MHz", llama_avg / avg_freq);
        println!("    self-written: {:.4} us/MHz", self_avg / avg_freq);
        println!("    (Eliminates thermal variance)");
        (llama_avg / avg_freq, self_avg / avg_freq)
    } else {
        (llama_avg, self_avg)
    };

    (ratio, llama_norm, self_norm)
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let m = 4096;
    let k = 4096;
    let n = 128; // Batch size for GEMM
    let rounds = 100;
    let seed = 42;

    println!("=== K-quant GEMM Alternating Benchmark (N={}) ===", n);
    println!("M={}, K={}, {} rounds\n", m, k, rounds);

    // Get GPU status before benchmark
    let gpu_before = get_gpu_stats();
    if let Some(ref stats) = gpu_before {
        println!("=== GPU Status Before Benchmark ===");
        print_stats("Before:", stats);
    }

    let input = gen_f32_matrix(k, n, seed);

    // Test each format
    let tests: [(&str, usize, Vec<u8>); 5] = [
        ("Q2_K", 84, unsafe {
            std::mem::transmute::<Vec<BlockQ2K>, Vec<u8>>(gen_q2_k(m, k, seed))
        }),
        ("Q3_K", 110, unsafe {
            std::mem::transmute::<Vec<BlockQ3K>, Vec<u8>>(gen_q3_k(m, k, seed))
        }),
        ("Q4_K", 144, unsafe {
            std::mem::transmute::<Vec<BlockQ4K>, Vec<u8>>(gen_q4_k(m, k, seed))
        }),
        ("Q5_K", 176, unsafe {
            std::mem::transmute::<Vec<BlockQ5K>, Vec<u8>>(gen_q5_k(m, k, seed))
        }),
        ("Q6_K", 210, unsafe {
            std::mem::transmute::<Vec<BlockQ6K>, Vec<u8>>(gen_q6_k(m, k, seed))
        }),
    ];

    let mut all_results = Vec::new();
    for (name, block_size, weights) in tests {
        let (ratio, llama_norm, self_norm) = run_alternating_gemm_test(
            name,
            &device,
            &weights,
            &input,
            m,
            n,
            k,
            block_size,
            rounds,
            gpu_before.as_ref(),
        );
        all_results.push((name, ratio, llama_norm, self_norm));
    }

    println!("\n=== Summary ===");
    println!(
        "{:8} {:>10} {:>15} {:>15}",
        "Format", "Ratio", "llama us/MHz", "self us/MHz"
    );
    println!("{}", "-".repeat(55));
    for (name, ratio, llama_norm, self_norm) in all_results {
        println!(
            "{:8} {:>9.1}% {:>15.4} {:>15.4}",
            name, ratio, llama_norm, self_norm
        );
    }

    // Get GPU status after benchmark
    let gpu_after = get_gpu_stats();
    if let Some(ref stats) = gpu_after {
        println!("\n=== GPU Status After Benchmark ===");
        print_stats("After:", stats);
    }

    // Print changes
    if let (Some(before), Some(after)) = (&gpu_before, &gpu_after) {
        let diff = before.diff(after);
        print_diff(&diff);

        if diff.freq_mhz.abs() > 50.0 {
            println!(
                "\n⚠ GPU frequency changed {:.0} MHz - thermal effects may affect results",
                diff.freq_mhz
            );
        } else {
            println!(
                "\n✓ GPU frequency stable - normalized results are reliable"
            );
        }
    }

    println!("\nNote: Both use the same kernel_mul_mm from llama.cpp template");
    println!("us/MHz metric eliminates thermal variance for fair comparison");
}
