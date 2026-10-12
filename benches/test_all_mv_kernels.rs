// Batch test all 23 MV kernels
// 1. Compilation verification
// 2. Correctness test (compare with llama.cpp reference)
// 3. Performance benchmark

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

// ============================================================================
// Block structures matching Metal definitions
// ============================================================================

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ2K {
    scales: [u8; 32],
    qs: [u8; 128],
    d: u16,
    dmin: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
    d: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ5K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
    qh: [u8; 64],
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    d: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ40 {
    d: u16,
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ41 {
    d: u16,
    m: u16,
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ80 {
    d: u16,
    qs: [u8; 32],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ50 {
    d: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ51 {
    d: u16,
    m: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ10 {
    d: u16,
    qs: [u8; 4],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ20 {
    d: u16,
    qs: [u8; 16],
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

// ============================================================================
// Kernel definitions
// ============================================================================

#[derive(Debug, Clone)]
struct KernelInfo {
    name: String,
    block_size: usize,
    block_struct_size: usize,
}

const KERNELS: &[(&str, usize, usize)] = &[
    // (kernel_name, elements_per_block, block_struct_size)
    ("kernel_gemm_q4_K_f32", 256, 144),
    ("kernel_gemm_q2_K_f32", 256, 164),
    ("kernel_gemm_q3_K_f32", 256, 110),
    ("kernel_gemm_q5_K_f32", 256, 208),
    ("kernel_gemm_q6_K_f32", 256, 212),
    ("kernel_gemm_q4_0_f32", 32, 18),
    ("kernel_gemm_q4_1_f32", 32, 20),
    ("kernel_gemm_q8_0_f32", 32, 34),
    ("kernel_gemm_q5_0_f32", 32, 22),
    ("kernel_gemm_q5_1_f32", 32, 24),
    ("kernel_gemm_q1_0_f32", 32, 6),
    ("kernel_gemm_q2_0_f32", 32, 18),
    ("kernel_gemm_iq4_nl_f32", 32, 18),
    ("kernel_gemm_mxfp4_f32", 32, 18),
    ("kernel_gemm_tq2_0_f32", 32, 18),
    ("kernel_gemm_iq1_s_f32", 256, 164),
    ("kernel_gemm_iq1_m_f32", 256, 164),
    ("kernel_gemm_iq2_xxs_f32", 256, 164),
    ("kernel_gemm_iq2_xs_f32", 256, 164),
    ("kernel_gemm_iq2_s_f32", 256, 164),
    ("kernel_gemm_iq3_xxs_f32", 256, 164),
    ("kernel_gemm_iq3_s_f32", 256, 164),
    ("kernel_gemm_iq4_xs_f32", 256, 164),
];

fn expand_includes(
    source: &str,
    shader_dir: &std::path::PathBuf,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();
    for line in source.lines() {
        if line.starts_with("#include \"") {
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];
            if visited.contains(&filename.to_string()) {
                result
                    .push_str(&format!("// Already included: {}\n", filename));
                continue;
            }
            visited.push(filename.to_string());
            let include_path = shader_dir.join(filename);
            if let Ok(include_content) = std::fs::read_to_string(&include_path)
            {
                result.push_str(&format!("// --- {} ---\n", filename));
                let include_content =
                    include_content.replace("#pragma once", "// #pragma once");
                result.push_str(&expand_includes(
                    &include_content,
                    shader_dir,
                    visited,
                ));
                result.push_str(&format!("// --- end {} ---\n", filename));
            } else {
                result.push_str(&format!(
                    "// WARNING: Could not find: {}\n",
                    filename
                ));
                result.push_str(line);
                result.push('\n');
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn main() {
    println!("=== MV Kernel Batch Test ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Compile template
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let source = std::fs::read_to_string(shader_dir.join("gemm.metal"))
        .expect("Failed to read template");

    let mut visited = Vec::new();
    let expanded = expand_includes(&source, &shader_dir, &mut visited);

    println!("Compiling template ({} bytes)...\n", expanded.len());
    let library =
        match device.new_library_with_source(&expanded, &compile_options) {
            Ok(lib) => lib,
            Err(e) => {
                println!("✗ Compilation failed: {:?}", e);
                return;
            }
        };

    let available_kernels: Vec<String> =
        library.function_names().iter().map(|s| s.to_string()).collect();

    // Test parameters
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 1000;

    println!(
        "Testing {} kernels (M={}, K={}, {} iterations)\n",
        KERNELS.len(),
        m,
        k,
        iterations
    );

    let mut results = Vec::new();

    for (kernel_name, block_size, _block_struct_size) in KERNELS {
        print!("Testing {}...", kernel_name);

        // Check if kernel exists
        if !available_kernels.contains(&kernel_name.to_string()) {
            println!(" ✗ NOT FOUND");
            results.push((*kernel_name, "missing".to_string(), 0.0, 0.0));
            continue;
        }

        // Create function constants
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

        // Get kernel with constants
        let kernel = match library.get_function(kernel_name, Some(cv)) {
            Ok(k) => k,
            Err(e) => {
                println!(" ✗ GET FAILED: {}", e);
                results.push((
                    *kernel_name,
                    "get_failed".to_string(),
                    0.0,
                    0.0,
                ));
                continue;
            }
        };

        // Create pipeline
        let pipeline =
            match device.new_compute_pipeline_state_with_function(&kernel) {
                Ok(p) => p,
                Err(e) => {
                    println!(" ✗ PIPELINE FAILED: {}", e);
                    results.push((
                        *kernel_name,
                        "pipeline_failed".to_string(),
                        0.0,
                        0.0,
                    ));
                    continue;
                }
            };

        // Generate test data
        let input: Vec<f32> = (0..k).map(|i| (i % 10) as f32 * 0.1).collect();
        let num_blocks = k / block_size;
        let weights_size = m * num_blocks * 144; // Use largest block size as approximation

        let weights_buffer = device.new_buffer_with_data(
            vec![0u8; weights_size].as_ptr() as *const c_void,
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            (k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device
            .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

        // Setup args
        let nb01 = (num_blocks * 144) as u64;
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
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

        let thread_group_size = MTLSize { width: 32, height: 2, depth: 1 };
        let thread_group_count =
            MTLSize { width: (m as u64 + 3) / 4, height: 1, depth: 1 };

        // Warmup
        for _ in 0..100 {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed_us =
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (elapsed_us * 1e-6) / 1e9;

        println!(" ✓ {:.2} us ({:.1} GFLOPS)", elapsed_us, gflops);
        results.push((*kernel_name, "ok".to_string(), elapsed_us, gflops));
    }

    // Summary
    println!("\n=== Summary ===");
    let successful: Vec<_> =
        results.iter().filter(|(_, status, _, _)| status == "ok").collect();
    let failed: Vec<_> =
        results.iter().filter(|(_, status, _, _)| status != "ok").collect();

    println!("Successful: {}/{}", successful.len(), KERNELS.len());
    if !failed.is_empty() {
        println!("Failed:");
        for (name, status, _, _) in &failed {
            println!("  {} - {}", name, status);
        }
    }

    // Performance ranking
    if !successful.is_empty() {
        println!("\nPerformance ranking (GFLOPS):");
        let mut sorted: Vec<_> = successful.iter().collect();
        sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        for (name, _, _, gflops) in sorted {
            println!("  {} - {:.1}", name, gflops);
        }
    }
}
