// Self-written Q4_K GPU test
// Measures CPU time and GPU time separately
// Run: cargo run --release --bin rust_q4_k_fair

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const FC_MUL_MV: NSUInteger = 600;

#[repr(C)]
struct MulMVArgs {
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

fn create_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 0,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 2,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 3,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 4,
        );
    }
    cv
}

fn main() {
    println!("=== Self-written Q4_K GPU Test ===");

    let m = 1024;
    let k = 4096;
    let nb = k / 256;
    let block_size = 144;
    let iterations = 10000;

    // STEP 1: Generate data
    let t1 = Instant::now();
    let weights_size = m * nb * block_size;
    let weights: Vec<u8> =
        (0..weights_size).map(|i| ((i * 17 + 13) % 256) as u8).collect();
    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();
    let t2 = Instant::now();
    println!(
        "Step 1 - Generate data: {} µs",
        t2.duration_since(t1).as_micros()
    );

    // STEP 2: Create GPU buffers
    let t3 = Instant::now();
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const _,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let t4 = Instant::now();
    println!(
        "Step 2 - Create buffers: {} µs",
        t4.duration_since(t3).as_micros()
    );

    // STEP 3: Load kernel
    let t5 = Instant::now();
    let shader = std::fs::read_to_string(
        "/Users/jia/Desktop/infer_train/shaders/mul_mv_template.metal",
    )
    .unwrap();
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    let library =
        device.new_library_with_source(&shader, &compile_options).unwrap();
    let cv = create_function_constants(2, 1, 1, 1);
    let kernel =
        library.get_function("kernel_mul_mv_q4_K_f32", Some(cv)).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();
    let t6 = Instant::now();
    println!(
        "Step 3 - Compile kernel: {} µs",
        t6.duration_since(t5).as_micros()
    );

    // Args
    let args = MulMVArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: (nb * block_size) as u64,
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
        nr0: 2,
        r2: 1,
        r3: 1,
    };

    // Warmup
    let grid = MTLSize::new((m / 4) as u64, 1, 1);
    let tg = MTLSize::new(64, 1, 1);

    for _ in 0..100 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // Benchmark
    println!("\nBenchmarking {} iterations...", iterations);
    let bench_start = Instant::now();

    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    let bench_end = Instant::now();
    let total_us = bench_end.duration_since(bench_start).as_micros() as f64;
    let avg_us = total_us / iterations as f64;

    println!("\n=== Results ===");
    println!("Total time: {} µs", total_us);
    println!("Avg time: {:.2} µs", avg_us);
    println!("GFLOPS: {:.1}", (2.0 * m as f64 * k as f64) / avg_us / 1000.0);

    // Verify output
    let result = unsafe {
        std::slice::from_raw_parts(output_buf.contents() as *const f32, m)
    };
    println!(
        "\nFirst 5 results: {:.4} {:.4} {:.4} {:.4} {:.4}",
        result[0], result[1], result[2], result[3], result[4]
    );
}
