// Alternating benchmark for all K-quant formats
// Uses correct block structures and alternating execution to eliminate thermal bias

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

struct KQuantFormat {
    name: &'static str,
    nsg: i16,
    nr0: i32,
    block_size: usize,
}

fn run_alternating_benchmark(
    device: &Device,
    queue: &CommandQueue,
    format: &KQuantFormat,
    llama_lib: &Library,
    my_lib: &Library,
    m: usize,
    k: usize,
    rounds: usize,
) -> (f64, f64, usize) {
    let cv = create_function_constants(format.nsg, 1, 1, 1);

    let kernel_name = format!("kernel_gemm_{}_f32", format.name);
    let llama_func =
        llama_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();
    let my_func = my_lib.get_function(&kernel_name, Some(cv)).unwrap();

    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    let nb = k / 256;
    let weights_size = m * nb * format.block_size;
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
    let llama_output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let my_output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    let args = MulMVArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: (nb * format.block_size) as u64,
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
        nr0: format.nr0,
        r2: 1,
        r3: 1,
    };

    let grid = MTLSize::new(
        ((m + (format.nsg as usize * format.nr0 as usize) - 1)
            / (format.nsg as usize * format.nr0 as usize)) as u64,
        1,
        1,
    );
    let tg = MTLSize::new((format.nsg as u64) * 32, 1, 1);

    // Warmup
    for _ in 0..20 {
        for (pipeline, output_buf) in [
            (&llama_pipeline, &llama_output_buf),
            (&my_pipeline, &my_output_buf),
        ]
        .iter()
        {
            let cmd_buf = queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MulMVArgs>() as u64,
                &args as *const _ as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buf), 0);
            encoder.set_buffer(2, Some(&input_buf), 0);
            encoder.set_buffer(3, Some(output_buf), 0);
            encoder.dispatch_thread_groups(grid, tg);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
    }

    // Alternating benchmark
    let mut llama_times = Vec::new();
    let mut my_times = Vec::new();

    for _ in 0..rounds {
        // Run llama.cpp
        let start = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&llama_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        llama_times.push(start.elapsed().as_nanos() as f64);

        // Run self-written
        let start = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&my_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&my_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        my_times.push(start.elapsed().as_nanos() as f64);
    }

    // Calculate medians
    llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    my_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let llama_median = llama_times[rounds / 2];
    let my_median = my_times[rounds / 2];

    // Count wins
    let mut wins = 0;
    for i in 0..rounds {
        if my_times[i] < llama_times[i] {
            wins += 1;
        }
    }

    (llama_median, my_median, wins)
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load shaders
    let my_shader = std::fs::read_to_string(
        "/Users/jia/Desktop/infer_trasrc/ops/metal/gemm.metal",
    )
    .unwrap();

    // Create libraries
    let compile_options = CompileOptions::new();
    let my_lib =
        device.new_library_with_source(&my_shader, &compile_options).unwrap();

    // For llama.cpp, we use the same shader since code is identical
    let llama_lib =
        device.new_library_with_source(&my_shader, &compile_options).unwrap();

    let m = 1024;
    let k = 4096;
    let rounds = 500;

    let formats = [
        KQuantFormat { name: "q2_K", nsg: 2, nr0: 2, block_size: 84 },
        KQuantFormat { name: "q3_K", nsg: 2, nr0: 2, block_size: 110 },
        KQuantFormat { name: "q4_K", nsg: 2, nr0: 2, block_size: 144 },
        KQuantFormat { name: "q5_K", nsg: 2, nr0: 2, block_size: 176 },
        KQuantFormat { name: "q6_K", nsg: 2, nr0: 2, block_size: 210 },
    ];

    println!("=== K-quant MV Alternating Benchmark ===");
    println!("M={}, K={}, {} rounds, alternating execution", m, k, rounds);
    println!(
        "Using same shader for both (code is identical after threadgroup fix)\n"
    );
    println!(
        "{:8} | {:>12} | {:>12} | {:>6} | {:>12}",
        "Format", "llama.cpp", "Self-wr", "Ratio", "Wins/Total"
    );
    println!("{}", "-".repeat(60));

    for format in &formats {
        let (llama_time, my_time, wins) = run_alternating_benchmark(
            &device, &queue, format, &llama_lib, &my_lib, m, k, rounds,
        );

        let ratio = (my_time / llama_time) * 100.0;
        let llama_us = llama_time / 1000.0;
        let my_us = my_time / 1000.0;

        let status = if ratio < 100.0 {
            "✓ FASTER"
        } else if ratio > 102.0 {
            "SLOWER"
        } else {
            "TIED"
        };
        println!(
            "{:8} | {:>10.2} µs | {:>10.2} µs | {:>5.1}% | {:>4}/{:<4} {}",
            format.name, llama_us, my_us, ratio, wins, rounds, status
        );
    }

    println!(
        "\nNote: Since both use identical shader code, any performance difference is due to thermal/GPU variance."
    );
}
