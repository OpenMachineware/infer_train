// Fair comparison: my implementation vs llama.cpp original
// Loads two separate Metal libraries and runs alternating tests

use half::f16;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_diff, print_stats};

const QK: usize = 32;

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

fn bench_vs_llama(
    name: &str,
    llama_kernel: &str,
    self_kernel: &str,
    device: &Device,
    llama_lib: &metal::Library,
    self_lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    args: &GemmArgs,
    m: usize,
    n: usize,
    rounds: usize,
    gpu_freq: f64,
) {
    let queue = device.new_command_queue();

    // Function constants
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

    // Create pipelines
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib.get_function(llama_kernel, Some(cv.clone())).unwrap(),
        )
        .unwrap();

    let self_pipeline = device
        .new_compute_pipeline_state_with_function(
            &self_lib.get_function(self_kernel, Some(cv)).unwrap(),
        )
        .unwrap();

    // Buffers
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

    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_self = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Threadgroup size: 32 threads in x, 4 simdgroups = 128 threads total
    // Simdgroup kernel: r0 = tgpig.y * NR0 (rows), r1 = tgpig.x * NR1 (cols)
    // NR0 = 64, NR1 = 32
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((n + 31) / 32) as u64, // x: column tiles (NR1 = 32)
        ((m + 63) / 64) as u64, // y: row tiles (NR0 = 64)
        1,
    );

    // Warmup
    for _ in 0..10 {
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
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
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

    // Alternating test
    let mut llama_total = 0u64;
    let mut self_total = 0u64;
    let mut self_wins = 0usize;

    println!("\n{} (llama.cpp vs self-written):", name);
    print!("  ");

    for i in 0..rounds {
        // llama.cpp
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
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let llama_us = start.elapsed().as_micros() as u64;
        llama_total += llama_us;

        // Self-written
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let self_us = start.elapsed().as_micros() as u64;
        self_total += self_us;

        if self_us < llama_us {
            self_wins += 1;
        }

        if (i + 1) % 20 == 0 {
            print!(".");
        }
    }
    println!();

    let llama_avg = llama_total as f64 / rounds as f64;
    let self_avg = self_total as f64 / rounds as f64;
    let ratio = self_avg / llama_avg * 100.0;

    println!(
        "  llama.cpp: {:.0} µs ({:.3} µs/MHz)",
        llama_avg,
        llama_avg / gpu_freq
    );
    println!(
        "  self:     {:.0} µs ({:.3} µs/MHz)",
        self_avg,
        self_avg / gpu_freq
    );
    println!("  Ratio:    {:.1}% (self/llama.cpp)", ratio);
    println!("  Self wins: {}/{}", self_wins, rounds);
}

fn main() {
    println!("=== GEMM Fair Comparison: llama.cpp vs self-written ===");

    let device = Device::system_default().expect("No Metal device");

    let m = 4096usize;
    let k = 4096usize;
    let n = 128usize;
    let rounds = 100usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}, {} rounds\n", m, k, n, rounds);

    // GPU status
    let gpu_before = get_gpu_stats();
    let gpu_freq = if let Some(ref g) = gpu_before {
        print_stats("=== GPU Status ===", g);
        g.gpu_freq_mhz as f64
    } else {
        388.0
    };

    // Load llama.cpp library (combined file with inlined dependencies)
    let llama_lib_source =
        include_str!("../../shaders/mul_mm_llama_combined.metal");
    let compile_opts = metal::CompileOptions::new();
    let llama_lib =
        match device.new_library_with_source(llama_lib_source, &compile_opts) {
            Ok(lib) => {
                println!("✓ llama.cpp library compiled");
                lib
            }
            Err(e) => {
                println!("✗ llama.cpp library failed to compile:");
                println!("  {}", e);
                println!("\nFalling back to self-written only test...");
                std::process::exit(1);
            }
        };

    // Load self-written library
    let self_lib_source = include_str!("../../shaders/mul_mm_standalone.metal");
    let self_lib = device
        .new_library_with_source(self_lib_source, &compile_opts)
        .expect("Failed to compile self library");
    println!("✓ self-written library compiled");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    println!("\n=== Standard Q GEMM ===");

    // Test formats: (name, llama_kernel, self_kernel, block_size)
    let formats: [(&str, &str, &str, usize); 5] = [
        ("Q4_0", "kernel_mul_mm_q4_0_f32", "kernel_mul_mm_q4_0_f32", 18),
        ("Q8_0", "kernel_mul_mm_q8_0_f32", "kernel_mul_mm_q8_0_f32", 34),
        ("Q4_1", "kernel_mul_mm_q4_1_f32", "kernel_mul_mm_q4_1_f32", 20),
        ("Q5_0", "kernel_mul_mm_q5_0_f32", "kernel_mul_mm_q5_0_f32", 22),
        ("Q5_1", "kernel_mul_mm_q5_1_f32", "kernel_mul_mm_q5_1_f32", 24),
    ];

    for (name, llama_kernel, self_kernel, block_size) in formats.iter() {
        let nb = k / QK;
        let mut weights = Vec::with_capacity(m * nb * block_size);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..m * nb * block_size {
            weights.push(rng.r#gen::<u8>());
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            ne03: 1,
            nb00: *block_size as u64,
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

        bench_vs_llama(
            name,
            llama_kernel,
            self_kernel,
            &device,
            &llama_lib,
            &self_lib,
            &weights,
            &input_f16,
            &args,
            m,
            n,
            rounds,
            gpu_freq,
        );
    }

    println!("\n=== MXFP4 and IQ4_NL GEMM ===");

    let special: [(&str, &str, &str, usize); 2] = [
        ("MXFP4", "kernel_mul_mm_mxfp4_f32", "kernel_mul_mm_mxfp4_f32", 17),
        ("IQ4_NL", "kernel_mul_mm_iq4_nl_f32", "kernel_mul_mm_iq4_nl_f32", 18),
    ];

    for (name, llama_kernel, self_kernel, block_size) in special.iter() {
        let nb = k / QK;
        let mut weights = Vec::with_capacity(m * nb * block_size);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..m * nb * block_size {
            weights.push(rng.r#gen::<u8>());
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            ne03: 1,
            nb00: *block_size as u64,
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

        bench_vs_llama(
            name,
            llama_kernel,
            self_kernel,
            &device,
            &llama_lib,
            &self_lib,
            &weights,
            &input_f16,
            &args,
            m,
            n,
            rounds,
            gpu_freq,
        );
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
