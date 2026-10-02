// Compare nsg=2 (llama.cpp default) vs optimized nsg values
// Alternating benchmark for fair comparison

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

fn benchmark_pipeline(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    name: &str,
    nsg: i16,
    nr0: i32,
    block_size: usize,
    m: usize,
    k: usize,
) -> f64 {
    let cv = create_function_constants(nsg, 1, 1, 1);
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

    // Alternating benchmark
    const ITERATIONS: usize = 200;
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
    let lib =
        device.new_library_with_source(shader, &CompileOptions::new()).unwrap();

    let m = 1024;
    let k = 4096;

    println!("=== NSG Optimization: Default vs Optimized ===");
    println!("M={}, K={}, 200 iterations alternating\n", m, k);
    println!(
        "{:8} | {:>10} | {:>10} | {:>6} | {:>6}",
        "Format", "nsg=2 (µs)", "opt (µs)", "ratio", "improvement"
    );
    println!("{}", "-".repeat(60));

    // Test configurations: (name, nr0, block_size, optimal_nsg)
    let configs = [
        ("q2_K", 4, 164, 4), // 11% faster with nsg=4
        ("q3_K", 2, 110, 1), // slightly better with nsg=1
        ("q4_K", 2, 144, 4), // 0.7% faster with nsg=4
        ("q5_K", 1, 176, 8), // best with nsg=8
        ("q6_K", 2, 210, 2), // nsg=2 is already optimal
    ];

    for (name, nr0, block_size, opt_nsg) in configs {
        let time_nsg2 = benchmark_pipeline(
            &device, &queue, &lib, name, 2, nr0, block_size, m, k,
        );
        let time_opt = benchmark_pipeline(
            &device, &queue, &lib, name, opt_nsg, nr0, block_size, m, k,
        );

        let ratio = time_opt / time_nsg2 * 100.0;
        let improvement = (time_nsg2 - time_opt) / time_nsg2 * 100.0;

        let status = if ratio < 100.0 { "✓" } else { "" };
        println!(
            "{:8} | {:>10.2} | {:>10.2} | {:>5.1}% | {:+.2}% {}",
            name,
            time_nsg2 / 1000.0,
            time_opt / 1000.0,
            ratio,
            improvement,
            status
        );
    }
}
