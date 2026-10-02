// GPU timing benchmark - eliminates CPU-side variance
// Uses Metal GPU timestamps for accurate measurement

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::thread;
use std::time::{Duration, Instant};

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

fn run_gpu_time_benchmark(
    device: &Device,
    queue: &CommandQueue,
    info: &KQuantInfo,
    llama_shader: &str,
    my_shader: &str,
    m: usize,
    k: usize,
    iterations: usize,
) -> (f64, f64) {
    let cv = create_function_constants(info.nsg, 1, 1, 1);

    let compile_options = CompileOptions::new();
    let llama_lib =
        device.new_library_with_source(llama_shader, &compile_options).unwrap();
    let my_lib =
        device.new_library_with_source(my_shader, &compile_options).unwrap();

    let kernel_name = format!("kernel_mul_mv_{}_f32", info.name);
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
    let output_buf = device
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

    // Warmup
    for _ in 0..20 {
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
        encoder.set_buffer(3, Some(&output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // Measure llama.cpp using GPU time
    let mut llama_gpu_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let start = cmd_buf.gpu_start_time();

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
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

        let end = cmd_buf.gpu_end_time();
        let gpu_time_ns = (end - start) as f64;
        llama_gpu_times.push(gpu_time_ns);
    }

    // Small cooldown
    thread::sleep(Duration::from_secs(5));

    // Warmup for self-written
    for _ in 0..20 {
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
        encoder.set_buffer(3, Some(&output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // Measure self-written using GPU time
    let mut my_gpu_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let start = cmd_buf.gpu_start_time();

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&my_pipeline);
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

        let end = cmd_buf.gpu_end_time();
        let gpu_time_ns = (end - start) as f64;
        my_gpu_times.push(gpu_time_ns);
    }

    llama_gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    my_gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let llama_median = llama_gpu_times[iterations / 2];
    let my_median = my_gpu_times[iterations / 2];

    (llama_median, my_median)
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let llama_shader = std::fs::read_to_string(
        "/Users/jia/Desktop/infer_train/llama.cpp-0.5.0/build/bin/kernels/mul_mv.metal"
    ).unwrap();
    let my_shader = std::fs::read_to_string(
        "/Users/jia/Desktop/infer_train/shaders/mul_mv_template.metal",
    )
    .unwrap();

    let m = 1024;
    let k = 4096;

    let formats = [
        KQuantInfo { name: "q2_K", nsg: 2, nr0: 2, block_size: 96 },
        KQuantInfo { name: "q3_K", nsg: 2, nr0: 2, block_size: 128 },
        KQuantInfo { name: "q4_K", nsg: 2, nr0: 2, block_size: 144 },
        KQuantInfo { name: "q5_K", nsg: 2, nr0: 2, block_size: 176 },
        KQuantInfo { name: "q6_K", nsg: 2, nr0: 2, block_size: 210 },
    ];

    println!("GPU Time Benchmark ({} iterations)", 100);
    println!("Format   | llama.cpp (µs) | self-written (µs) | Ratio");
    println!("---------|----------------|-------------------|-------");

    for info in &formats {
        let (llama_time, my_time) = run_gpu_time_bench(
            &device,
            &queue,
            info,
            &llama_shader,
            &my_shader,
            m,
            k,
            100,
        );

        let ratio = (my_time / llama_time) * 100.0;
        let llama_us = llama_time / 1000.0;
        let my_us = my_time / 1000.0;

        println!(
            "{:8} | {:>14.2} | {:>17.2} | {:>6.1}%",
            info.name, llama_us, my_us, ratio
        );
    }
}
