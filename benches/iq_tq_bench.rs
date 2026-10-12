// Benchmark IQ/TQ/MXFP4 MV kernels vs llama.cpp

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

struct FormatInfo {
    name: &'static str,
    nsg: i16,
    nr0: i32,
    block_size: usize,
}

fn run_benchmark(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    info: &FormatInfo,
    m: usize,
    k: usize,
) -> f64 {
    let cv = create_function_constants(info.nsg, 1, 1, 1);
    let kernel_name = format!("kernel_mul_mm_{}_f32", info.name);
    let func = match lib.get_function(&kernel_name, Some(cv)) {
        Ok(f) => f,
        Err(_) => {
            println!("  {} - kernel not found", info.name);
            return 0.0;
        }
    };
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

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

    // Compile template with includes
    let shader_base = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/blocks.h");
    // Define GGML_COMMON_IMPL_METAL before including blocks.h
    let ggml_common_with_metal =
        "#define GGML_COMMON_IMPL_METAL\n".to_string() + ggml_common;
    let shader =
        shader_base.replace("#include \"blocks.h\"", &ggml_common_with_metal);

    let compile_options = CompileOptions::new();
    let lib = match device.new_library_with_source(&shader, &compile_options) {
        Ok(l) => l,
        Err(e) => {
            println!("Shader compilation failed: {}", e);
            return;
        }
    };

    let m = 1024;
    let k = 4096;

    println!("=== IQ/TQ/MXFP4 MV Performance ===");
    println!("M={}, K={}, 100 iterations median\n", m, k);
    println!("{:12} | {:>10} | {:>8}", "Format", "Time (µs)", "Status");
    println!("{}", "-".repeat(35));

    // IQ/TQ/MXFP4 formats
    let formats = [
        FormatInfo { name: "iq1_s", nsg: 4, nr0: 4, block_size: 32 },
        FormatInfo { name: "iq1_m", nsg: 4, nr0: 4, block_size: 40 },
        FormatInfo { name: "iq2_xxs", nsg: 4, nr0: 4, block_size: 64 },
        FormatInfo { name: "iq2_xs", nsg: 4, nr0: 4, block_size: 80 },
        FormatInfo { name: "iq2_s", nsg: 4, nr0: 4, block_size: 96 },
        FormatInfo { name: "iq3_xxs", nsg: 4, nr0: 4, block_size: 80 },
        FormatInfo { name: "iq3_s", nsg: 4, nr0: 4, block_size: 96 },
        FormatInfo { name: "iq4_nl", nsg: 2, nr0: 2, block_size: 36 },
        FormatInfo { name: "iq4_xs", nsg: 2, nr0: 2, block_size: 144 },
        FormatInfo { name: "tq1_0", nsg: 4, nr0: 4, block_size: 8 },
        FormatInfo { name: "tq2_0", nsg: 4, nr0: 4, block_size: 16 },
        FormatInfo { name: "mxfp4", nsg: 2, nr0: 2, block_size: 34 },
    ];

    for info in formats {
        let median_ns = run_benchmark(&device, &queue, &lib, &info, m, k);
        if median_ns > 0.0 {
            println!("{:12} | {:>10.2} | OK", info.name, median_ns / 1000.0);
        }
    }
}
