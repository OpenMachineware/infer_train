// Alternating IQ kernel benchmark vs llama.cpp
// A-B-A-B pattern eliminates thermal bias

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const FC_MUL_MV: NSUInteger = 600;

fn create_function_constants(nsg: i16) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 0,
        );
        let ne12: i16 = 1;
        let r2: i16 = 1;
        let r3: i16 = 1;
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

struct IQFormatInfo {
    name: &'static str,
    nsg: i16,
    block_size: usize,
}

fn run_alternating(
    device: &Device,
    queue: &CommandQueue,
    info: &IQFormatInfo,
    llama_lib: &Library,
    my_lib: &Library,
    m: usize,
    k: usize,
    rounds: usize,
) -> (f64, f64, usize, usize) {
    let cv = create_function_constants(info.nsg);
    // llama.cpp uses kernel_mul_mv_*, we use kernel_gemm_*
    let llama_kernel_name = format!("kernel_mul_mv_{}_f32", info.name);
    let my_kernel_name = format!("kernel_gemm_{}_f32", info.name);

    let llama_func =
        match llama_lib.get_function(&llama_kernel_name, Some(cv.clone())) {
            Ok(f) => f,
            Err(_) => return (0.0, 0.0, 0, 0),
        };
    let my_func = match my_lib.get_function(&my_kernel_name, Some(cv)) {
        Ok(f) => f,
        Err(_) => return (0.0, 0.0, 0, 0),
    };

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
    let llama_out = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let my_out = device
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
        nr0: 4,
        r2: 1,
        r3: 1,
    };

    let grid = MTLSize::new(((m + 127) / 128) as u64, 1, 1);
    let tg = MTLSize::new((info.nsg as u64) * 32, 1, 1);

    // Warmup
    for _ in 0..20 {
        for (pipeline, out) in
            [(&llama_pipeline, &llama_out), (&my_pipeline, &my_out)]
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MulMVArgs>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(out), 0);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    // Alternating A-B-A-B
    let mut llama_times = Vec::with_capacity(rounds);
    let mut my_times = Vec::with_capacity(rounds);
    let mut llama_wins = 0;
    let mut my_wins = 0;

    for _ in 0..rounds {
        let t1 = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&llama_out), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let llama_time = t1.elapsed().as_nanos() as f64;

        let t2 = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&my_out), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let my_time = t2.elapsed().as_nanos() as f64;

        llama_times.push(llama_time);
        my_times.push(my_time);

        if my_time < llama_time {
            my_wins += 1;
        } else {
            llama_wins += 1;
        }
    }

    // Calculate medians
    llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    my_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let llama_median = llama_times[rounds / 2];
    let my_median = my_times[rounds / 2];
    (llama_median, my_median, llama_wins, my_wins)
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load llama.cpp shader
    // The combined file already has ggml-metal-impl.h and common.h content inline
    // but needs ggml-common.h for block types (block_q1_0, etc.)
    let llama_shader_base =
        std::fs::read_to_string("/tmp/combined_llama.metal")
            .expect("llama.cpp shader not found at /tmp/combined_llama.metal");
    let ggml_common = std::fs::read_to_string("/tmp/ggml-common.h")
        .expect("ggml-common.h not found");

    // Strip includes and pragmas from ggml-common.h
    let ggml_common_clean: String = ggml_common
        .lines()
        .filter(|l| {
            !l.trim().starts_with("#pragma once")
                && !l.trim().starts_with("#include")
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Add Metal definitions and ggml-common.h content at the beginning
    let llama_shader = format!(
        "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\nusing namespace metal;\n{}\n{}",
        ggml_common_clean, llama_shader_base
    );

    // Remove all include and pragma lines, and embed placeholders
    let llama_shader = llama_shader
        .lines()
        .filter(|line| {
            let l = line.trim();
            !l.starts_with("#pragma once")
                && !l.starts_with("#include")
                && !l.contains("__embed_ggml-common.h__")
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Load and inline self-written shader
    let my_shader_base = include_str!("../src/ops/metal/gemm.metal");
    let my_ggml_common = include_str!("../src/ops/metal/ggml-common.h");

    // Build complete shader with all tables from ggml-common.h
    let ggml_common_with_metal =
        format!("#define GGML_COMMON_IMPL_METAL\n{}", my_ggml_common);
    let my_shader = my_shader_base.replace("#include \"ggml-common.h\"", "");
    let my_shader = format!("{}\n{}", ggml_common_with_metal, my_shader);

    let compile_options = CompileOptions::new();
    let llama_lib = device
        .new_library_with_source(&llama_shader, &compile_options)
        .unwrap();
    let my_lib =
        device.new_library_with_source(&my_shader, &compile_options).unwrap();

    let m = 1024;
    let k = 4096;
    let rounds = 500;

    let formats = [
        IQFormatInfo { name: "iq1_s", nsg: 4, block_size: 36 },
        IQFormatInfo { name: "iq1_m", nsg: 4, block_size: 36 },
        IQFormatInfo { name: "iq2_xxs", nsg: 4, block_size: 66 },
        IQFormatInfo { name: "iq2_xs", nsg: 4, block_size: 70 },
        IQFormatInfo { name: "iq2_s", nsg: 4, block_size: 78 },
        IQFormatInfo { name: "iq3_xxs", nsg: 4, block_size: 66 },
        IQFormatInfo { name: "iq3_s", nsg: 4, block_size: 74 },
        IQFormatInfo { name: "iq4_nl", nsg: 4, block_size: 18 },
        IQFormatInfo { name: "iq4_xs", nsg: 2, block_size: 42 },
    ];

    println!("=== IQ MV Alternating Benchmark ===");
    println!("M={}, K={}, {} rounds", m, k, rounds);
    println!();
    println!(
        "{:12} | {:>10} | {:>10} | {:>5} | Status",
        "Format", "llama(µs)", "self(µs)", "Wins"
    );
    println!("{}", "-".repeat(55));

    for info in &formats {
        let (llama_time, my_time, llama_wins, my_wins) = run_alternating(
            &device, &queue, info, &llama_lib, &my_lib, m, k, rounds,
        );

        if llama_time == 0.0 {
            println!(
                "{:12} | {:>10} | {:>10} | {:>5} | MISSING",
                info.name, "-", "-", "-"
            );
            continue;
        }

        let llama_us = llama_time / 1000.0;
        let my_us = my_time / 1000.0;
        let ratio = my_time / llama_time * 100.0;

        let status = if my_wins > llama_wins {
            format!("FASTER ({:.1}%)", ratio)
        } else if my_wins < llama_wins {
            format!("SLOWER ({:.1}%)", ratio)
        } else {
            format!("TIED ({:.1}%)", ratio)
        };

        println!(
            "{:12} | {:>10.2} | {:>10.2} | {:>3}/{:<3} | {}",
            info.name, llama_us, my_us, my_wins, rounds, status
        );
    }
}
