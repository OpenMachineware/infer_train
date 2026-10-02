// Alternating benchmark for K-quant MV kernels
// Eliminates thermal bias by alternating between llama.cpp and self-written
use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
    d: u16,
}

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
    let mut skip_embed_block = false;
    let mut skip_endif_count = 0;
    let mut after_embed_else = false;
    // Track when we've passed #else in an embed block - need to skip the matching #endif

    for line in source.lines() {
        // Handle embed library block
        if line.contains("GGML_METAL_EMBED_LIBRARY") {
            skip_embed_block = true;
            skip_endif_count = 1;
            result.push_str("// Skipping embed block\n");
            continue;
        }
        if skip_embed_block {
            if line.starts_with("#if") {
                skip_endif_count += 1;
            } else if line.starts_with("#else") {
                // When we hit #else in an embed block, stop skipping
                // The #else branch contains the actual include we want
                skip_embed_block = false;
                after_embed_else = true;
                result
                    .push_str("// End of embed block, start of actual code\n");
            } else if line.starts_with("#endif") {
                skip_endif_count -= 1;
                if skip_endif_count == 0 {
                    skip_embed_block = false;
                }
            }
            continue;
        }
        if after_embed_else && line.starts_with("#endif") {
            // Skip the #endif that closes the embed block
            after_embed_else = false;
            result.push_str("// Skipped embed #endif\n");
            continue;
        }

        if line.starts_with("#include \"") {
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];
            if !visited.contains(&filename.to_string()) {
                visited.push(filename.to_string());
                // Search in multiple directories (ggml-common.h is 2 levels up from kernels)
                let search_paths = vec![
                    base_dir.join(filename),
                    base_dir.parent().unwrap().join(filename),
                    base_dir.parent().unwrap().parent().unwrap().join(filename),
                    base_dir
                        .parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .join(filename),
                ];
                let mut found = false;
                for include_path in &search_paths {
                    if let Ok(content) = std::fs::read_to_string(include_path) {
                        // Use the directory of the included file as the new base
                        let new_base = include_path.parent().unwrap();
                        result.push_str(&expand_includes(
                            &content.replace("#pragma once", "// #pragma once"),
                            new_base,
                            visited,
                        ));
                        found = true;
                        break;
                    }
                }
                if !found {
                    // If file not found, keep the include line commented
                    result.push_str("// #include \"");
                    result.push_str(filename);
                    result.push_str("\" // NOT FOUND in ");
                    for p in &search_paths {
                        result.push_str(&format!("{:?}, ", p));
                    }
                    result.push('\n');
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
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Load llama.cpp kernel library (with includes expanded)
    let llama_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("llama.cpp-0.5.0/ggml/src/ggml-metal/kernels");
    let llama_raw =
        std::fs::read_to_string(llama_dir.join("mul_mv.metal")).unwrap();
    let llama_source = expand_includes(
        &llama_raw,
        &llama_dir,
        &mut vec!["mul_mv.metal".to_string()],
    );
    let llama_lib = device
        .new_library_with_source(&llama_source, &compile_options)
        .unwrap();

    // Load self-written template
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source =
        std::fs::read_to_string(shader_dir.join("gemm.metal")).unwrap();
    let my_lib =
        device.new_library_with_source(&my_source, &compile_options).unwrap();

    let m = 4096usize;
    let k = 4096usize;
    let alternating_runs = 100;
    let iterations_per_run = 100;

    // Test Q4_K and Q3_K
    for (name, block_size) in [("Q4_K", 144), ("Q3_K", 110)] {
        println!("\n=== {} Alternating Benchmark ===", name);
        println!(
            "M={}, K={}, {} runs x {} iterations",
            m, k, alternating_runs, iterations_per_run
        );

        let kernel_name = format!("kernel_gemm_{}_f32", name.to_lowercase());
        let qk = 256;
        let nb = k / qk;
        let nb01 = (block_size * nb) as u64;

        let weights_size = m * nb * block_size;
        let weights: Vec<u8> =
            (0..weights_size).map(|i| (i % 256) as u8).collect();
        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let input: Vec<f32> = (0..k).map(|i| (i % 10) as f32 * 0.1).collect();
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: block_size as u64,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let cv = create_cv();
        let output_size = m * 4;
        let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
        let tg_count =
            MTLSize { width: (m as u64 + 3) / 4, height: 1, depth: 1 };

        let llama_pipeline = device
            .new_compute_pipeline_state_with_function(
                &llama_lib
                    .get_function(&kernel_name, Some(cv.clone()))
                    .unwrap(),
            )
            .unwrap();
        let my_pipeline = device
            .new_compute_pipeline_state_with_function(
                &my_lib.get_function(&kernel_name, Some(cv.clone())).unwrap(),
            )
            .unwrap();

        let llama_output = device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let my_output = device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        for _ in 0..50 {
            for pipeline in [&llama_pipeline, &my_pipeline] {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipeline);
                enc.set_bytes(
                    0,
                    std::mem::size_of::<MvArgs>() as u64,
                    &args as *const MvArgs as *const c_void,
                );
                enc.set_buffer(1, Some(&weights_buffer), 0);
                enc.set_buffer(2, Some(&input_buffer), 0);
                enc.set_buffer(3, Some(&llama_output), 0);
                enc.dispatch_thread_groups(tg_count, tg_size);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }

        // Alternating benchmark
        let mut llama_times = Vec::with_capacity(alternating_runs);
        let mut my_times = Vec::with_capacity(alternating_runs);

        for run in 0..alternating_runs {
            // Run llama.cpp FIRST
            let mut llama_time = 0u128;
            for _ in 0..iterations_per_run {
                let start = Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&llama_pipeline);
                enc.set_bytes(
                    0,
                    std::mem::size_of::<MvArgs>() as u64,
                    &args as *const MvArgs as *const c_void,
                );
                enc.set_buffer(1, Some(&weights_buffer), 0);
                enc.set_buffer(2, Some(&input_buffer), 0);
                enc.set_buffer(3, Some(&llama_output), 0);
                enc.dispatch_thread_groups(tg_count, tg_size);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                llama_time += start.elapsed().as_micros();
            }
            llama_times.push(llama_time as f64 / iterations_per_run as f64);

            // Run self-written SECOND
            let mut my_time = 0u128;
            for _ in 0..iterations_per_run {
                let start = Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&my_pipeline);
                enc.set_bytes(
                    0,
                    std::mem::size_of::<MvArgs>() as u64,
                    &args as *const MvArgs as *const c_void,
                );
                enc.set_buffer(1, Some(&weights_buffer), 0);
                enc.set_buffer(2, Some(&input_buffer), 0);
                enc.set_buffer(3, Some(&my_output), 0);
                enc.dispatch_thread_groups(tg_count, tg_size);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                my_time += start.elapsed().as_micros();
            }
            my_times.push(my_time as f64 / iterations_per_run as f64);

            if (run + 1) % 20 == 0 {
                println!(
                    "Run {}/{}: llama={:.2}us, self={:.2}us, ratio={:.1}%",
                    run + 1,
                    alternating_runs,
                    llama_times.last().unwrap(),
                    my_times.last().unwrap(),
                    llama_times.last().unwrap() / my_times.last().unwrap()
                        * 100.0
                );
            }
        }

        // Calculate median
        llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        my_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let llama_median = llama_times[alternating_runs / 2];
        let my_median = my_times[alternating_runs / 2];
        let ratio = llama_median / my_median * 100.0;

        println!("\nMedian results:");
        println!("  llama.cpp:   {:.2} us", llama_median);
        println!("  self-written: {:.2} us", my_median);
        println!("  Ratio:        {:.1}% of llama.cpp", ratio);

        // Check correctness
        let llama_ptr = llama_output.contents() as *const f32;
        let my_ptr = my_output.contents() as *const f32;
        let correct: usize = (0..m)
            .filter(|&i| {
                let lv = unsafe { *llama_ptr.add(i) };
                let mv = unsafe { *my_ptr.add(i) };
                if lv.abs() > 1e-6 {
                    (lv - mv).abs() / lv.abs() < 0.01
                } else {
                    (lv - mv).abs() < 1e-6
                }
            })
            .count();
        println!(
            "  Correctness: {}/{} ({:.1}%)",
            correct,
            m,
            correct as f32 / m as f32 * 100.0
        );
    }
}
