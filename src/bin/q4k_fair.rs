// Fair alternating benchmark: Q4_K MV self-written vs llama.cpp
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C)]
struct MvArgs {
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

fn create_cv() -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let nsg: i16 = 2;
        let ne12: i16 = 1;
        let r2: i16 = 1;
        let r3: i16 = 1;
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn expand_includes(
    source: &str,
    base_dir: &std::path::Path,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();
    for line in source.lines() {
        if line.starts_with("#include \"") {
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];
            if !visited.contains(&filename.to_string()) {
                visited.push(filename.to_string());
                let include_path = base_dir.join(filename);
                if let Ok(content) = std::fs::read_to_string(&include_path) {
                    result.push_str(&expand_includes(
                        &content.replace("#pragma once", ""),
                        base_dir,
                        visited,
                    ));
                }
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);

    // Load llama.cpp kernel with includes expanded
    let llama_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("llama.cpp-0.6.0/ggml/src/ggml-metal/kernels");
    let llama_raw =
        std::fs::read_to_string(llama_dir.join("mul_mv.metal")).unwrap();
    let llama_source = expand_includes(
        &llama_raw,
        &llama_dir,
        &mut vec!["mul_mv.metal".to_string()],
    );
    let llama_lib = device
        .new_library_with_source(&llama_source, &opts)
        .expect("Failed to compile llama");

    // Load our kernel
    let our_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let grid_tables =
        std::fs::read_to_string(our_dir.join("iq_grid_tables.h")).unwrap();
    let iq_extra = r#"
constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
constexpr constant static int8_t ksigns_iq2xs[128] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,
     1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,
     1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,
     1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
};
"#;
    let our_raw = std::fs::read_to_string(our_dir.join("gemm.metal")).unwrap();
    let our_source = format!("{}\n{}\n{}", iq_extra, grid_tables, our_raw);
    let our_lib = device
        .new_library_with_source(&our_source, &opts)
        .expect("Failed to compile ours");

    let cv = create_cv();
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib
                .get_function("kernel_mul_mv_q4_K_f32", Some(cv.clone()))
                .unwrap(),
        )
        .unwrap();
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(
            &our_lib.get_function("kernel_gemm_q4_K_f32", Some(cv)).unwrap(),
        )
        .unwrap();

    // Test data
    let m = 4096usize;
    let k = 4096usize;
    let block_size = 144;
    let nb = k / 256;
    let mut weights = vec![0u8; m * nb * block_size];
    for i in 0..weights.len() {
        weights[i] = ((i * 17) % 256) as u8;
    }
    let mut input = vec![0.0f32; k];
    for i in 0..k {
        input[i] = (i as f32 % 10.0) * 0.1;
    }

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let llama_out = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let our_out = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: block_size as u64,
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

    let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
    let tg_count = MTLSize { width: ((m + 1) / 2) as u64, height: 1, depth: 1 };

    // Warmup
    for _ in 0..50 {
        for pipeline in [&llama_pipeline, &our_pipeline] {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&llama_out), 0);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    // Alternating benchmark
    let rounds = 200;
    let mut llama_times = Vec::new();
    let mut our_times = Vec::new();

    for _ in 0..rounds {
        // llama.cpp
        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&llama_out), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        llama_times.push(start.elapsed().as_micros() as f64);

        // Ours
        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&our_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&our_out), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        our_times.push(start.elapsed().as_micros() as f64);
    }

    let avg_llama: f64 = llama_times.iter().sum::<f64>() / rounds as f64;
    let avg_ours: f64 = our_times.iter().sum::<f64>() / rounds as f64;
    let ratio = avg_ours / avg_llama * 100.0;
    let wins = (0..rounds).filter(|&i| our_times[i] < llama_times[i]).count();

    println!(
        "=== Q4_K MV Alternating (M={}, K={}, {} rounds) ===",
        m, k, rounds
    );
    println!("llama.cpp: {:.1} us", avg_llama);
    println!("Ours:      {:.1} us", avg_ours);
    println!(
        "Ratio:     {:.1}% ({})",
        ratio,
        if ratio < 100.0 {
            "FASTER"
        } else if ratio < 101.0 {
            "TIED"
        } else {
            "SLOWER"
        }
    );
    println!("Wins:      {}/{}", wins, rounds);
}
