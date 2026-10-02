// Precise timing breakdown for MV kernel
// Measure each step separately to find bottlenecks

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const FC_MUL_MV: NSUInteger = 600;

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

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Q6_K shader (both llama.cpp and self-written use same code)
    let shader = include_str!("../../shaders/mul_mv_template.metal");

    let compile_options = CompileOptions::new();
    let lib = device
        .new_library_with_source(shader, &compile_options)
        .expect("Compilation failed");

    let cv = create_function_constants(2, 1, 1, 1);
    let func = lib.get_function("kernel_mul_mv_q6_K_f32", Some(cv)).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    let m = 1024;
    let k = 4096;
    let nb = k / 256;
    let block_size = 210; // Q6_K block size

    // ========================================
    // STEP 1: CPU data preparation timing
    // ========================================
    println!("=== Step-by-Step Timing Breakdown ===\n");

    let t1 = Instant::now();
    let weights_size = m * nb * block_size;
    let mut weights_data = vec![0u8; weights_size];
    for i in 0..weights_size {
        weights_data[i] = ((i * 17 + 13) % 256) as u8;
    }
    let cpu_generate_time = t1.elapsed();
    println!("Step 1 - CPU generate random bytes: {:?}", cpu_generate_time);

    // ========================================
    // STEP 2: GPU buffer allocation timing
    // ========================================
    let t2 = Instant::now();
    let weights_buf = device
        .new_buffer(weights_size as u64, MTLResourceOptions::StorageModeShared);
    let input_buf = device
        .new_buffer((k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let buffer_alloc_time = t2.elapsed();
    println!("Step 2 - GPU buffer allocation: {:?}", buffer_alloc_time);

    // ========================================
    // STEP 3: CPU → GPU transfer timing (write weights)
    // ========================================
    let t3 = Instant::now();
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights_data.as_ptr(),
            weights_buf.contents() as *mut u8,
            weights_size,
        );
    }
    let weights_transfer_time = t3.elapsed();
    println!("Step 3 - CPU→GPU write weights: {:?}", weights_transfer_time);

    // ========================================
    // STEP 4: CPU → GPU transfer timing (write input)
    // ========================================
    let t4 = Instant::now();
    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr(),
            input_buf.contents() as *mut f32,
            k,
        );
    }
    let input_transfer_time = t4.elapsed();
    println!("Step 4 - CPU→GPU write input: {:?}", input_transfer_time);

    // ========================================
    // STEP 5: GPU kernel execution timing
    // ========================================
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
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let grid = MTLSize::new(256, 1, 1);
    let tg = MTLSize::new(64, 1, 1);

    // Warmup
    for _ in 0..10 {
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

    // Measure kernel execution
    let mut cpu_wait_times = Vec::new();

    for _ in 0..100 {
        let cmd_buf = queue.new_command_buffer();

        let t_start = Instant::now();
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
        let cpu_time = t_start.elapsed();
        cpu_wait_times.push(cpu_time.as_nanos() as f64);
    }

    cpu_wait_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_cpu_wait = cpu_wait_times[50];

    println!(
        "Step 5 - GPU kernel execution (CPU wait): {:.2} us",
        median_cpu_wait / 1000.0
    );

    // ========================================
    // STEP 6: GPU → CPU read timing
    // ========================================
    let t6 = Instant::now();
    let output: &[f32] = unsafe {
        std::slice::from_raw_parts(output_buf.contents() as *const f32, m)
    };
    let _sum: f64 = output.iter().map(|&x| x as f64).sum(); // Force read
    let read_time = t6.elapsed();
    println!("Step 6 - GPU→CPU read output: {:?}", read_time);

    // ========================================
    // Summary
    // ========================================
    println!("\n=== Total Time Breakdown ===");
    let total_cpu_time = cpu_generate_time
        + buffer_alloc_time
        + weights_transfer_time
        + input_transfer_time;
    println!("CPU preparation total: {:?}", total_cpu_time);
    println!("GPU execution median: {:.2} us", median_cpu_wait / 1000.0);
    println!("GPU read back: {:?}", read_time);

    // ========================================
    // Now measure llama.cpp for comparison
    // ========================================
    println!("\n=== Comparing with llama.cpp (same kernel code) ===");
}
