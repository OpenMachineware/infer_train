// Standalone benchmark for Q3_K MV kernel - self-written version
// Run: cargo run --release --bin self_q3k_bench

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
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

fn main() {
    println!("=== Self-written Q3_K MV Benchmark ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Load self-written template
    let shader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/ops/metal/gemm.metal");
    let source =
        std::fs::read_to_string(&shader_path).expect("Failed to read shader");
    let library = device
        .new_library_with_source(&source, &compile_options)
        .expect("Failed to compile shader");

    // Create pipeline with function constants
    let cv = create_cv();
    let kernel = library
        .get_function("kernel_gemm_q3_K_f32", Some(cv))
        .expect("Failed to get kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    println!(
        "Pipeline created. maxThreadsPerThreadgroup: {}",
        pipeline.max_threads_per_threadgroup().width
    );

    // Test dimensions (same as C++ version)
    const M: usize = 4096;
    const K: usize = 4096;
    const QK: usize = 256;
    const NB: usize = K / QK;
    const BLOCK_SIZE: usize = 110;

    // Generate weights (same deterministic pattern as C++)
    let mut weights: Vec<BlockQ3K> =
        vec![
            BlockQ3K { hmask: [0; 32], qs: [0; 64], scales: [0; 12], d: 0 };
            M * NB
        ];
    for i in 0..(M * NB) {
        for j in 0..32 {
            weights[i].hmask[j] = ((i * 17 + j * 23) % 256) as u8;
        }
        for j in 0..64 {
            weights[i].qs[j] = ((i * 13 + j * 7) % 256) as u8;
        }
        for j in 0..12 {
            weights[i].scales[j] = ((i * 31 + j * 11) % 128) as u8;
        }
        weights[i].d = ((0.5f32 + (i % 10) as f32 * 0.1) * 1.0) as u16; // half bits
    }

    // Generate input (same deterministic pattern as C++)
    let input: Vec<f32> = (0..K).map(|i| (i % 100) as f32 * 0.01).collect();

    // Create buffers
    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (M * NB * BLOCK_SIZE) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (K * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device
        .new_buffer((M * 4) as u64, MTLResourceOptions::StorageModeShared);

    // Args
    let args = MvArgs {
        ne00: K as i32,
        ne01: M as i32,
        ne02: 1,
        nb00: BLOCK_SIZE as u64,
        nb01: (NB * BLOCK_SIZE) as u64,
        nb02: 0,
        nb03: 0,
        ne10: K as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (K * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: M as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };
    let args_buf = device.new_buffer_with_data(
        &args as *const MvArgs as *const c_void,
        std::mem::size_of::<MvArgs>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Thread configuration (same as C++ version)
    const NSG: u64 = 2;
    const NR0: u64 = 2;
    let threadgroups =
        MTLSize::new((M as u64 + NSG * NR0 - 1) / (NSG * NR0), 1, 1);
    let threads_per_tg = MTLSize::new(32, NSG, 1);

    // Warmup
    println!("Warming up...");
    for _ in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&args_buf), 0);
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_buf), 0);
        enc.dispatch_thread_groups(threadgroups, threads_per_tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    const ITERATIONS: usize = 10000;
    println!("Running benchmark ({} iterations)...", ITERATIONS);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&args_buf), 0);
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_buf), 0);
        enc.dispatch_thread_groups(threadgroups, threads_per_tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = start.elapsed();
    let total_us = elapsed.as_micros() as f64;
    let avg_us = total_us / ITERATIONS as f64;
    let gflops = 2.0 * M as f64 * K as f64 / (avg_us * 1e-6) / 1e9;

    println!("\nResults:");
    println!("  Total time: {:.2} ms", total_us / 1000.0);
    println!("  Avg time:   {:.2} us", avg_us);
    println!("  GFLOPS:     {:.1f}", gflops);

    // Print first few output values for verification
    let output = unsafe {
        std::slice::from_raw_parts(output_buf.contents() as *const f32, M)
    };
    println!("\nFirst 5 outputs:");
    for i in 0..5 {
        println!("  output[{}] = {:.6}", i, output[i]);
    }
}
