// True alternating test: nsg=2 vs optimized nsg
// A-B-A-B in a single run to eliminate thermal variance

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

struct TestConfig {
    name: &'static str,
    nr0: i32,
    block_size: usize,
    opt_nsg: i16,
}

fn run_alternating_test(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    config: &TestConfig,
    m: usize,
    k: usize,
) -> (f64, f64) {
    // Create two pipelines: nsg=2 and opt_nsg
    let cv_nsg2 = create_function_constants(2, 1, 1, 1);
    let cv_opt = create_function_constants(config.opt_nsg, 1, 1, 1);

    let kernel_name = format!("kernel_gemm_{}_f32", config.name);
    let func_nsg2 = lib.get_function(&kernel_name, Some(cv_nsg2)).unwrap();
    let func_opt = lib.get_function(&kernel_name, Some(cv_opt)).unwrap();

    let pipeline_nsg2 =
        device.new_compute_pipeline_state_with_function(&func_nsg2).unwrap();
    let pipeline_opt =
        device.new_compute_pipeline_state_with_function(&func_opt).unwrap();

    let nb = k / 256;
    let weights_size = m * nb * config.block_size;
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
    let out_nsg2 = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let out_opt = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    let args = MulMVArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: (nb * config.block_size) as u64,
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

    let grid_nsg2 = MTLSize::new(
        ((m + (2 * config.nr0 as usize) - 1) / (2 * config.nr0 as usize))
            as u64,
        1,
        1,
    );
    let tg_nsg2 = MTLSize::new(64, 1, 1);

    let grid_opt = MTLSize::new(
        ((m + (config.opt_nsg as usize * config.nr0 as usize) - 1)
            / (config.opt_nsg as usize * config.nr0 as usize)) as u64,
        1,
        1,
    );
    let tg_opt = MTLSize::new((config.opt_nsg as u64) * 32, 1, 1);

    // Warmup both
    for _ in 0..10 {
        for (pipeline, grid, tg, output) in [
            (&pipeline_nsg2, grid_nsg2, tg_nsg2, &out_nsg2),
            (&pipeline_opt, grid_opt, tg_opt, &out_opt),
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
            encoder.set_buffer(3, Some(output), 0);
            encoder.dispatch_thread_groups(*grid, *tg);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
    }

    // True alternating: A-B-A-B
    const ITERATIONS: usize = 200;
    let mut times_nsg2 = Vec::with_capacity(ITERATIONS);
    let mut times_opt = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        // A: nsg=2
        let start = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_nsg2);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&out_nsg2), 0);
        encoder.dispatch_thread_groups(grid_nsg2, tg_nsg2);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        times_nsg2.push(start.elapsed().as_nanos() as f64);

        // B: opt_nsg
        let start = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_opt);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&out_opt), 0);
        encoder.dispatch_thread_groups(grid_opt, tg_opt);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        times_opt.push(start.elapsed().as_nanos() as f64);
    }

    times_nsg2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_opt.sort_by(|a, b| a.partial_cmp(b).unwrap());

    (times_nsg2[ITERATIONS / 2], times_opt[ITERATIONS / 2])
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let shader = include_str!("../../shaders/gemm.metal");
    let lib =
        device.new_library_with_source(shader, &CompileOptions::new()).unwrap();

    let m = 1024;
    let k = 4096;

    println!("=== True Alternating: nsg=2 vs Optimized ===");
    println!("M={}, K={}, 200 iterations A-B-A-B\n", m, k);
    println!(
        "{:8} | {:>10} | {:>10} | {:>6} | {:>6}",
        "Format", "nsg=2 (µs)", "opt (µs)", "ratio", "improvement"
    );
    println!("{}", "-".repeat(60));

    let configs = [
        TestConfig { name: "q2_K", nr0: 4, block_size: 164, opt_nsg: 4 },
        TestConfig { name: "q3_K", nr0: 2, block_size: 110, opt_nsg: 1 },
        TestConfig { name: "q4_K", nr0: 2, block_size: 144, opt_nsg: 4 },
        TestConfig { name: "q5_K", nr0: 1, block_size: 176, opt_nsg: 8 },
        TestConfig { name: "q6_K", nr0: 2, block_size: 210, opt_nsg: 2 }, // nsg=2 is already optimal
    ];

    for config in configs {
        let (time_nsg2, time_opt) =
            run_alternating_test(&device, &queue, &lib, &config, m, k);
        let ratio = time_opt / time_nsg2 * 100.0;
        let improvement = (time_nsg2 - time_opt) / time_nsg2 * 100.0;
        let status = if ratio < 100.0 { "✓" } else { "" };

        println!(
            "{:8} | {:>10.2} | {:>10.2} | {:>5.1}% | {:+.2}% {}",
            config.name,
            time_nsg2 / 1000.0,
            time_opt / 1000.0,
            ratio,
            improvement,
            status
        );
    }
}
