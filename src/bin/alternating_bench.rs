// Alternating benchmark - eliminates thermal bias
// Runs A-B-A-B pattern to ensure both kernels tested in same thermal state

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

struct KQuantInfo {
    name: &'static str,
    nsg: i16,
    nr0: i32,
    block_size: usize,
}

fn run_alternating_benchmark(
    device: &Device,
    queue: &CommandQueue,
    info: &KQuantInfo,
    llama_shader: &str,
    my_shader: &str,
    m: usize,
    k: usize,
    rounds: usize,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    let cv = create_function_constants(info.nsg, 1, 1, 1);

    let compile_options = CompileOptions::new();
    let llama_lib =
        device.new_library_with_source(llama_shader, &compile_options).unwrap();
    let my_lib =
        device.new_library_with_source(my_shader, &compile_options).unwrap();

    let kernel_name = format!("kernel_gemm_{}_f32", info.name);
    let llama_func =
        llama_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();
    let my_func = my_lib.get_function(&kernel_name, Some(cv)).unwrap();

    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    let nb = k / 256;
    let weights_size = m * nb * info.block_size;
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
        nb01: (nb * info.block_size) as u64,
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
        ((m + (info.nsg as usize * info.nr0 as usize) - 1)
            / (info.nsg as usize * info.nr0 as usize)) as u64,
        1,
        1,
    );
    let tg = MTLSize::new((info.nsg as u64) * 32, 1, 1);

    // Warmup both
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

    // Alternating benchmark: A-B-A-B-...
    let mut llama_times = Vec::new();
    let mut my_times = Vec::new();

    for round in 0..rounds {
        // Run llama.cpp (A)
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

        // Run self-written (B)
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
    let mut llama_sorted = llama_times.clone();
    let mut my_sorted = my_times.clone();
    llama_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    my_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let llama_median = llama_sorted[rounds / 2];
    let my_median = my_sorted[rounds / 2];

    (llama_median, my_median, llama_times, my_times)
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let llama_shader =
        std::fs::read_to_string("/tmp/combined_llama.metal").unwrap();
    let my_shader = std::fs::read_to_string(
        "/Users/jia/Desktop/infer_train/shaders/gemm.metal",
    )
    .unwrap();

    let m = 1024;
    let k = 4096;
    let rounds = 200;

    let formats = [
        KQuantInfo { name: "q2_K", nsg: 2, nr0: 2, block_size: 96 },
        KQuantInfo { name: "q3_K", nsg: 2, nr0: 2, block_size: 128 },
        KQuantInfo { name: "q4_K", nsg: 2, nr0: 2, block_size: 144 },
        KQuantInfo { name: "q5_K", nsg: 2, nr0: 2, block_size: 176 },
        KQuantInfo { name: "q6_K", nsg: 2, nr0: 2, block_size: 210 },
    ];

    println!("Alternating Benchmark ({} rounds)", rounds);
    println!("Format   | llama.cpp (µs) | self-written (µs) | Ratio | Status");
    println!("---------|----------------|-------------------|-------|--------");

    for info in &formats {
        let (llama_time, my_time, llama_all, my_all) =
            run_alternating_benchmark(
                &device,
                &queue,
                info,
                &llama_shader,
                &my_shader,
                m,
                k,
                rounds,
            );

        let ratio = (my_time / llama_time) * 100.0;
        let llama_us = llama_time / 1000.0;
        let my_us = my_time / 1000.0;

        // Count how many times each is faster
        let mut llama_faster = 0;
        let mut my_faster = 0;
        for i in 0..rounds {
            if my_all[i] < llama_all[i] {
                my_faster += 1;
            } else {
                llama_faster += 1;
            }
        }

        let status = if ratio < 100.0 { "✓ FASTER" } else { "SLOWER" };
        println!(
            "{:8} | {:>14.2} | {:>17.2} | {:>6.1}% | {} ({} vs {})",
            info.name, llama_us, my_us, ratio, status, my_faster, llama_faster
        );
    }
}
