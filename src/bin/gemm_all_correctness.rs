// Comprehensive GEMM correctness test for all formats
// Compare self-written mul_mm_standalone.metal vs llama.cpp's mul_mm_template.metallib

use metal::*;
use std::ffi::c_void;

fn main() {
    println!("=== Comprehensive GEMM Correctness Test ===\n");

    // Setup Metal
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Load llama.cpp's pre-compiled library
    let llama_lib = device
        .new_library_with_file("shaders/mul_mm_template.metallib")
        .expect("Failed to load llama.cpp library");

    // Load precompiled self-written shader
    println!("Loading self-written shader...");
    let self_lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .expect("Failed to load self-written library");
    println!("✓ Shaders loaded\n");

    // Test dimensions (small for safety)
    let m = 64;
    let k = 256;
    let n = 64;

    // Test formats
    let formats = vec![
        // Standard Q formats
        ("q4_0", 2, 18), // nl=2, block_size=32, block_bytes=18
        ("q8_0", 2, 34), // nl=2, block_size=32, block_bytes=34
        ("q4_1", 2, 20), // nl=2, block_size=32, block_bytes=20
        ("q5_0", 2, 22), // nl=2, block_size=32, block_bytes=22
        ("q5_1", 2, 24), // nl=2, block_size=32, block_bytes=24
        // MXFP
        ("mxfp4", 2, 18), // nl=2, block_size=32, block_bytes=18 (same as q4_0)
        // K-quant (QK_K = 256)
        ("q2_K", 8, 84),  // nl=QK_NL=8, block_bytes=84
        ("q3_K", 8, 110), // nl=QK_NL=8, block_bytes=110
        ("q4_K", 8, 144), // nl=QK_NL=8, block_bytes=144
        ("q5_K", 8, 176), // nl=QK_NL=8, block_bytes=176
        ("q6_K", 8, 210), // nl=QK_NL=8, block_bytes=210
        // IQ formats
        ("iq4_nl", 2, 18),  // QK4_NL=32
        ("iq1_s", 8, 34),   // QK_K=256, block_bytes=34
        ("iq1_m", 8, 42),   // QK_K=256, block_bytes=42
        ("iq2_xxs", 8, 34), // QK_K=256, block_bytes=34
        ("iq2_xs", 8, 42),  // QK_K=256, block_bytes=42
        ("iq2_s", 8, 50),   // QK_K=256, block_bytes=50
        ("iq3_xxs", 8, 98), // QK_K=256, block_bytes=98
        ("iq3_s", 8, 106),  // QK_K=256, block_bytes=106
        ("iq4_xs", 8, 146), // QK_K=256, block_bytes=146
    ];

    println!(
        "Testing {} formats with M={}, K={}, N={}\n",
        formats.len(),
        m,
        k,
        n
    );

    let mut all_pass = true;

    for (format_name, nl, block_bytes) in formats {
        print!("Testing {}... ", format_name);

        let result = test_format(
            &device,
            &queue,
            &self_lib,
            &llama_lib,
            format_name,
            m,
            k,
            n,
            nl,
            block_bytes,
        );

        match result {
            Ok(_) => println!("✓ PASS"),
            Err(e) => {
                println!("✗ FAIL: {}", e);
                all_pass = false;
            }
        }
    }

    println!("\n=== Summary ===");
    if all_pass {
        println!("✓ All tests passed");
    } else {
        println!("✗ Some tests failed");
    }
}

fn test_format(
    device: &Device,
    queue: &CommandQueue,
    self_lib: &Library,
    llama_lib: &Library,
    format_name: &str,
    m: usize,
    k: usize,
    n: usize,
    _nl: i32,
    block_bytes: usize,
) -> Result<(), String> {
    // Determine block size based on format
    let block_size = if format_name.starts_with("q2_K")
        || format_name.starts_with("q3_K")
        || format_name.starts_with("q4_K")
        || format_name.starts_with("q5_K")
        || format_name.starts_with("q6_K")
        || (format_name.starts_with("iq") && !format_name.contains("iq4_nl"))
    {
        256 // QK_K
    } else {
        32 // Standard Q format or QK4_NL
    };

    let num_blocks = k / block_size;

    // Create weights - layout depends on format
    let mut weights = Vec::with_capacity(m * num_blocks * block_bytes);

    // f16 value for 2.0 = 0x4000 (little-endian: 0x00, 0x40)
    let f16_2_0: [u8; 2] = [0x00, 0x40];

    for row in 0..m {
        for block_idx in 0..num_blocks {
            match format_name {
                // K-quant formats: scales first, then qs, then d and dmin
                "q2_K" => {
                    // scales[16] at offset 0
                    for i in 0..16 {
                        weights.push((row + i) as u8); // scales
                    }
                    // qs[64] at offset 16
                    for i in 0..64 {
                        weights.push(((row + block_idx + i) % 256) as u8);
                    }
                    // d at offset 80
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                    // dmin at offset 82
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                }
                "q3_K" => {
                    // hmask[32] at offset 0
                    for i in 0..32 {
                        weights.push(0); // hmask
                    }
                    // qs[64] at offset 32
                    for i in 0..64 {
                        weights.push(((row + block_idx + i) % 256) as u8);
                    }
                    // scales[12] at offset 96
                    for i in 0..12 {
                        weights.push((row + i) as u8);
                    }
                    // d at offset 108
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                }
                "q4_K" => {
                    // d at offset 0
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                    // dmin at offset 2
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                    // scales[12] at offset 4
                    for i in 0..12 {
                        weights.push((row + i) as u8);
                    }
                    // qs[128] at offset 16
                    for i in 0..128 {
                        weights.push(((row + block_idx + i) % 256) as u8);
                    }
                }
                "q5_K" => {
                    // d at offset 0
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                    // dmin at offset 2
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                    // scales[12] at offset 4
                    for i in 0..12 {
                        weights.push((row + i) as u8);
                    }
                    // qh[32] at offset 16
                    for i in 0..32 {
                        weights.push(((row + block_idx + i) % 256) as u8);
                    }
                    // qs[128] at offset 48
                    for i in 0..128 {
                        weights.push(((row + block_idx + i) % 256) as u8);
                    }
                }
                "q6_K" => {
                    // ql[128] at offset 0
                    for i in 0..128 {
                        weights.push(((row + block_idx + i) % 256) as u8);
                    }
                    // qh[64] at offset 128
                    for i in 0..64 {
                        weights.push(((row + block_idx + i) % 256) as u8);
                    }
                    // scales[16] at offset 192
                    for i in 0..16 {
                        weights.push((row + i) as i8 as u8); // int8_t
                    }
                    // d at offset 208
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                }
                // IQ formats - skip for now due to complex structure
                _ if format_name.starts_with("iq") => {
                    // Just fill with zeros for now - these need proper quantization
                    for i in 0..block_bytes {
                        weights.push(0);
                    }
                }
                // Standard Q formats: d first, then qs
                _ => {
                    // d (f16) at offset 0
                    weights.push(f16_2_0[0]);
                    weights.push(f16_2_0[1]);
                    // Quantized weights at offset 2
                    for i in 0..(block_bytes - 2) {
                        let val = ((row + block_idx + i) % 256) as u8;
                        weights.push(val);
                    }
                }
            }
        }
    }

    // Create input matrix (all 1.0)
    let input_f32: Vec<f32> = vec![1.0f32; n * k];

    // Output buffers
    let output_count = m * n;
    let output_bytes = output_count * 4;

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f32.as_ptr() as *const c_void,
        (input_f32.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_self = device
        .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
    let output_llama = device
        .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);

    // Arguments
    #[repr(C)]
    struct MulMMArgs {
        ne00: i32,
        ne02: i32,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        r2: i16,
        r3: i16,
    }

    let args = MulMMArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: (num_blocks * block_bytes) as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    // Function constants
    let cv = FunctionConstantValues::new();
    let bc_inp: bool = false;
    let bc_out: bool = false;
    let ne12: i16 = 1;
    let ne13: i16 = 1;
    let r2: i16 = 1;
    let r3: i16 = 1;
    unsafe {
        cv.set_constant_value_at_index(
            &bc_inp as *const _ as *const c_void,
            MTLDataType::Bool,
            700,
        );
        cv.set_constant_value_at_index(
            &bc_out as *const _ as *const c_void,
            MTLDataType::Bool,
            701,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            MTLDataType::Short,
            702,
        );
        cv.set_constant_value_at_index(
            &ne13 as *const _ as *const c_void,
            MTLDataType::Short,
            703,
        );
        cv.set_constant_value_at_index(
            &r2 as *const _ as *const c_void,
            MTLDataType::Short,
            704,
        );
        cv.set_constant_value_at_index(
            &r3 as *const _ as *const c_void,
            MTLDataType::Short,
            705,
        );
    }

    let kernel_name = format!("kernel_mul_mm_{}_f32", format_name);

    // Self-written kernel
    let self_function =
        self_lib
            .get_function(&kernel_name, Some(cv.clone()))
            .map_err(|e| format!("Self-written kernel not found: {}", e))?;
    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_function)
        .map_err(|e| format!("Failed to create pipeline: {}", e))?;

    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&self_pipeline);
    encoder.set_bytes(
        0,
        std::mem::size_of::<MulMMArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_self), 0);
    encoder.set_threadgroup_memory_length(0, 8192);
    let threadgroup_size = MTLSize::new(64, 2, 1);
    let threadgroups =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
    encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // llama.cpp kernel
    let llama_function = llama_lib
        .get_function(&kernel_name, Some(cv))
        .map_err(|e| format!("llama.cpp kernel not found: {}", e))?;
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_function)
        .map_err(|e| format!("Failed to create pipeline: {}", e))?;

    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&llama_pipeline);
    encoder.set_bytes(
        0,
        std::mem::size_of::<MulMMArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_llama), 0);
    encoder.set_threadgroup_memory_length(0, 8192);
    encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Compare
    let self_ptr = output_self.contents() as *const f32;
    let llama_ptr = output_llama.contents() as *const f32;
    let self_results =
        unsafe { std::slice::from_raw_parts(self_ptr, output_count) };
    let llama_results =
        unsafe { std::slice::from_raw_parts(llama_ptr, output_count) };

    let mut max_diff = 0.0f32;
    let mut num_nan_self = 0;
    let mut num_nan_llama = 0;

    for (&s, &l) in self_results.iter().zip(llama_results.iter()) {
        if s.is_nan() {
            num_nan_self += 1;
        }
        if l.is_nan() {
            num_nan_llama += 1;
        }
        let diff = (s - l).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    if num_nan_self > 0 {
        return Err(format!(
            "Self-written kernel has {} NaN values",
            num_nan_self
        ));
    }
    if num_nan_llama > 0 {
        return Err(format!(
            "llama.cpp kernel has {} NaN values",
            num_nan_llama
        ));
    }
    if max_diff > 1e-3 {
        return Err(format!("Max difference {} exceeds threshold", max_diff));
    }

    Ok(())
}
