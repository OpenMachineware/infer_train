// Alternating IQ GEMM benchmark: self-written gemm.metal vs llama.cpp mul_mm_template.metal
// A-B-A-B pattern eliminates thermal bias

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

fn gen_iq1_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u16>();
            }
            BlockIQ1S { d: f16::from_f32(1.0).to_bits(), qs, qh }
        })
        .collect()
}

fn gen_iq1_m(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1M> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u8; 16];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..16 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ1M { qs, qh, scales }
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

fn gen_iq2_xs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2XS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u16; 32];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u16>();
            }
            for j in 0..8 {
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ2XS { d: f16::from_f32(1.0).to_bits(), qs, scales }
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

fn gen_iq4_nl(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
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

// Parameter structure for self-written gemm.metal (matches ggml_metal_kargs_mul_mv)
#[repr(C)]
struct GemmArgsMv {
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

// Parameter structure for llama.cpp mul_mm_template.metal (matches ggml_metal_kargs_mul_mm)
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

fn bench_alternating(
    name: &str,
    my_kernel: &str,
    llama_kernel: &str,
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    args_mv: &GemmArgsMv,
    args_mm: &GemmArgsMm,
    m: usize,
    n: usize,
    nr0: usize, // Rows per simdgroup (from kernel template parameter)
    rounds: usize,
) {
    let queue = device.new_command_queue();

    // Function constants for mul_mm_standalone.metallib (same as llama.cpp, FC 700+)
    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv_my = metal::FunctionConstantValues::new();
    cv_my.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        FC_MUL_MM as u64, // FC_mul_mm_bc_inp = 700
    );
    cv_my.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        (FC_MUL_MM + 1) as u64, // FC_mul_mm_bc_out = 701
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 2) as u64, // FC_mul_mm_ne12 = 702
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 3) as u64, // FC_mul_mm_ne13 = 703
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 4) as u64, // FC_mul_mm_r2 = 704
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 5) as u64, // FC_mul_mm_r3 = 705
    );

    // Function constants for llama.cpp mul_mm_template.metal (same FC 700+)
    let cv_llama = metal::FunctionConstantValues::new();
    cv_llama.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        FC_MUL_MM as u64, // FC_mul_mm_bc_inp = 700
    );
    cv_llama.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        (FC_MUL_MM + 1) as u64, // FC_mul_mm_bc_out = 701
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 2) as u64, // FC_mul_mm_ne12 = 702
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 3) as u64, // FC_mul_mm_ne13 = 703
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 4) as u64, // FC_mul_mm_r2 = 704
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 5) as u64, // FC_mul_mm_r3 = 705
    );

    let my_fn = my_lib.get_function(my_kernel, Some(cv_my)).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    let llama_fn =
        llama_lib.get_function(llama_kernel, Some(cv_llama)).unwrap();
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

    // Threadgroup dispatch
    // Kernel expects: tgpig.x = row dimension (r0), tgpig.y = column dimension (r1)
    // Each threadgroup covers NSG * nr0 rows, where NSG=4 (simdgroups)
    let nsg = 4;
    let rows_per_tg = nsg * nr0;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((m + rows_per_tg - 1) / rows_per_tg) as u64, // X dimension = rows
        n as u64,                                     // Y dimension = columns
        1,
    );

    // Warmup
    for _ in 0..5 {
        // Self-written kernel also uses GemmArgsMm (ggml_metal_kargs_mul_mm)
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&my_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgsMm>() as u64,
                args_mm as *const _ as *const c_void,
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
        // llama.cpp kernel uses GemmArgsMm
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&llama_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgsMm>() as u64,
                args_mm as *const _ as *const c_void,
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

    // Verify correctness by comparing outputs
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
        // A: self-written (uses GemmArgsMv)
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            args_mm as *const _ as *const c_void,
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

        // B: llama.cpp (uses GemmArgsMm)
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            args_mm as *const _ as *const c_void,
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

        if (i + 1) % 50 == 0 {
            print!(".");
        }
    }
    println!();

    let my_avg = my_total as f64 / rounds as f64;
    let llama_avg = llama_total as f64 / rounds as f64;
    let ratio = my_avg / llama_avg * 100.0;

    // Get GPU frequency for normalization
    let gpu_now = get_gpu_stats();
    let freq_mhz = gpu_now.map(|g| g.gpu_freq_mhz as f64).unwrap_or(1.0);

    // Performance per MHz (eliminates thermal variance excuse)
    let my_per_mhz = my_avg / freq_mhz;
    let llama_per_mhz = llama_avg / freq_mhz;
    let ratio_normalized = my_per_mhz / llama_per_mhz * 100.0;

    println!("  {}:", name);
    println!(
        "    Self:    {:.2} µs ({:.3} µs/MHz @ {:.0} MHz)",
        my_avg, my_per_mhz, freq_mhz
    );
    println!(
        "    llama.cpp: {:.2} µs ({:.3} µs/MHz)",
        llama_avg, llama_per_mhz
    );
    println!("    Ratio:   {:.1}% ({} / {} wins)", ratio, my_wins, rounds);
    println!("    Freq-normalized ratio: {:.1}%", ratio_normalized);
}

fn main() {
    println!("=== IQ GEMM Alternating Benchmark ===");
    println!(
        "Comparing self-written gemm.metal vs llama.cpp mul_mm_template.metal"
    );

    let device = Device::system_default().expect("No Metal device");

    // Start with smaller dimensions to verify stability
    let m = 1024usize;
    let k = 1024usize;
    let n = 1usize; // N=1 for MV (matrix-vector) comparison
    let rounds = 100usize;
    let seed = 42u64;

    println!("M={}, K={}, N={} (MV mode), {} rounds\n", m, k, n, rounds);

    // GPU status
    let gpu_before = get_gpu_stats();
    if let Some(ref g) = gpu_before {
        print_stats("=== GPU Status ===", g);
    }

    // Load pre-compiled libraries
    // Use mul_mm_standalone.metallib for self-written kernels (verified correct via gemm_quant_test)
    let my_lib_path =
        std::path::Path::new("shaders/mul_mm_standalone.metallib");
    let my_lib = device
        .new_library_with_file(my_lib_path)
        .expect("Failed to load self-written GEMM library");

    // Load pre-compiled llama.cpp mul_mm_template.metallib
    // MetalPerformancePrimitives cannot be compiled at runtime, so we use offline compilation
    let llama_lib_path =
        std::path::Path::new("shaders/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load pre-compiled llama.cpp template library");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    println!("\n=== IQ Format GEMM Comparison (All 9 formats) ===");

    // Helper to create both argument structures
    let make_args = |k: usize,
                     m: usize,
                     n: usize,
                     block_size: usize,
                     block_bytes: usize,
                     nr0: i32|
     -> (GemmArgsMv, GemmArgsMm) {
        let args_mv = GemmArgsMv {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: block_bytes as u64,
            nb01: (k / block_size * block_bytes) as u64,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: n as i32,
            ne12: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            nr0,
            r2: 1,
            r3: 1,
        };
        let args_mm = GemmArgsMm {
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
        (args_mv, args_mm)
    };

    // IQ1_S (256-element block, 50 bytes, nr0=4)
    println!("\n--- IQ1_S ---");
    let weights = gen_iq1_s(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 50, 4);
    bench_alternating(
        "IQ1_S",
        "kernel_mul_mm_iq1_s_f32",
        "kernel_mul_mm_iq1_s_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        4,
        rounds,
    );

    // IQ1_M (256-element block, 56 bytes, nr0=4)
    println!("\n--- IQ1_M ---");
    let weights = gen_iq1_m(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 56, 4);
    bench_alternating(
        "IQ1_M",
        "kernel_mul_mm_iq1_m_f32",
        "kernel_mul_mm_iq1_m_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        4,
        rounds,
    );

    // IQ2_XXS (256-element block, 66 bytes, nr0=4)
    println!("\n--- IQ2_XXS ---");
    let weights = gen_iq2_xxs(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 66, 4);
    bench_alternating(
        "IQ2_XXS",
        "kernel_mul_mm_iq2_xxs_f32",
        "kernel_mul_mm_iq2_xxs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        4,
        rounds,
    );

    // IQ2_XS (256-element block, 74 bytes, nr0=4)
    println!("\n--- IQ2_XS ---");
    let weights = gen_iq2_xs(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 74, 4);
    bench_alternating(
        "IQ2_XS",
        "kernel_mul_mm_iq2_xs_f32",
        "kernel_mul_mm_iq2_xs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        4,
        rounds,
    );

    // IQ2_S (256-element block, 82 bytes, nr0=4)
    println!("\n--- IQ2_S ---");
    let weights = gen_iq2_s(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 82, 4);
    bench_alternating(
        "IQ2_S",
        "kernel_mul_mm_iq2_s_f32",
        "kernel_mul_mm_iq2_s_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        4,
        rounds,
    );

    // IQ3_XXS (256-element block, 98 bytes, nr0=4)
    println!("\n--- IQ3_XXS ---");
    let weights = gen_iq3_xxs(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 98, 4);
    bench_alternating(
        "IQ3_XXS",
        "kernel_mul_mm_iq3_xxs_f32",
        "kernel_mul_mm_iq3_xxs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        4,
        rounds,
    );

    // IQ3_S (256-element block, 110 bytes, nr0=4)
    println!("\n--- IQ3_S ---");
    let weights = gen_iq3_s(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 110, 4);
    bench_alternating(
        "IQ3_S",
        "kernel_mul_mm_iq3_s_f32",
        "kernel_mul_mm_iq3_s_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        4,
        rounds,
    );

    // IQ4_NL (32-element block, 18 bytes, nr0=2)
    println!("\n--- IQ4_NL ---");
    let weights = gen_iq4_nl(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 32, 18, 2);
    bench_alternating(
        "IQ4_NL",
        "kernel_mul_mm_iq4_nl_f32",
        "kernel_mul_mm_iq4_nl_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        2,
        rounds,
    );

    // IQ4_XS (256-element block, 136 bytes, nr0=2)
    println!("\n--- IQ4_XS ---");
    let weights = gen_iq4_xs(m, k, seed);
    let (args_mv, args_mm) = make_args(k, m, n, 256, 136, 2);
    bench_alternating(
        "IQ4_XS",
        "kernel_mul_mm_iq4_xs_f32",
        "kernel_mul_mm_iq4_xs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args_mv,
        &args_mm,
        m,
        n,
        2,
        rounds,
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
