// Sweep nsg values to find optimal threadgroup size

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

fn benchmark_nsg(
    device: &Device,
    queue: &CommandQueue,
    shader: &str,
    name: &str,
    nsg: i16,
    nr0: i32,
    block_size: usize,
    m: usize,
    k: usize,
) -> f64 {
    let cv = create_function_constants(nsg, 1, 1, 1);
    let lib =
        device.new_library_with_source(shader, &CompileOptions::new()).unwrap();
    let kernel_name = format!("kernel_gemm_{}_f32", name);
    let func = lib.get_function(&kernel_name, Some(cv)).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    let nb = k / 256;
    let weights_size = m * nb * block_size;
    let weights_data: Vec<u8> =
        (0..weights_size).map(|i| ((i * 17 + 13) % 256) as u8).collect();
    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

    let weights_buf = device.new_buffer_with_data(
        weights_data.as_ptr() as *const _,
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

    let grid = MTLSize::new(
        ((m + (nsg as usize * nr0 as usize) - 1)
            / (nsg as usize * nr0 as usize)) as u64,
        1,
        1,
    );
    let tg = MTLSize::new((nsg as u64) * 32, 1, 1);

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

    // Benchmark
    const ITERATIONS: usize = 100;
    let mut times = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
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
        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[ITERATIONS / 2]
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let shader = include_str!("../../shaders/gemm.metal");
    let m = 1024;
    let k = 4096;

    // Test configurations: (name, nr0, block_size)
    let formats = [
        ("q2_K", 4, 164),
        ("q3_K", 2, 110),
        ("q4_K", 2, 144),
        ("q5_K", 1, 176), // Fixed: was 208
        ("q6_K", 2, 210),
    ];

    println!("=== NSG Sweep for K-quant Kernels ===");
    println!("M={}, K={}, 100 iterations median\n", m, k);

    for (name, nr0, block_size) in formats {
        println!("\n--- {} (nr0={}, block={}B) ---", name, nr0, block_size);
        println!("{:>5} | {:>10} | {:>10}", "nsg", "time (µs)", "tg_size");

        for nsg in [1, 2, 4, 8].iter() {
            let median_ns = benchmark_nsg(
                &device, &queue, shader, name, *nsg, nr0, block_size, m, k,
            );
            let median_us = median_ns / 1000.0;
            let tg_size = nsg * 32;
            println!("{:5} | {:>10.2} | {:>10}", nsg, median_us, tg_size);
        }
    }
}
