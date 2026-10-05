// Simple GEMM correctness test
// Compare self-written mul_mm_standalone.metal vs llama.cpp's mul_mm_template.metallib

use metal::*;
use std::ffi::c_void;

fn main() {
    println!("=== GEMM Correctness Test ===");
    println!("Testing Q4_0 with small dimensions (M=32, K=256, N=32)\n");

    // Setup Metal
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Load llama.cpp's pre-compiled library
    let llama_lib = device
        .new_library_with_file("shaders/mul_mm_template.metallib")
        .expect("Failed to load llama.cpp library");
    println!("✓ Loaded llama.cpp library");

    // Load self-written pre-compiled library
    let self_lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .expect("Failed to load self-written library");
    println!("✓ Loaded self-written library\n");

    // Test dimensions
    let m = 64; // output rows
    let k = 256; // K dimension
    let n = 32; // output cols

    // Q4_0: block size = 32, each block = 2 bytes (scale) + 16 bytes (weights) = 18 bytes
    let block_size = 32;
    let block_bytes = 18;
    let num_blocks = k / block_size;

    // Create weights: non-symmetric pattern for non-zero results
    // Q4_0 dequant: nibble 0→-8, 1→-7, ..., 7→-1, 8→0, 9→1, ..., 15→7
    let mut weights = Vec::with_capacity(m * num_blocks * block_bytes);
    for row in 0..m {
        for block_idx in 0..num_blocks {
            // Scale = 2.0 (f16 = 0x4000)
            weights.push(0x00);
            weights.push(0x40);
            // 16 bytes of packed nibbles - use different high/low nibbles
            for i in 0..16 {
                let lo = ((row as u8 + block_idx as u8 + i) % 16) as u8;
                let hi = ((row as u8 + block_idx as u8 + i + 8) % 16) as u8;
                weights.push(lo | (hi << 4));
            }
        }
    }

    // Create input matrix B (f32): N rows, K columns, all 1.0 for simple calculation
    let input_f32: Vec<f32> = vec![1.0f32; n * k];

    // Output buffers
    let output_count = m * n;
    let output_bytes = output_count * 4;

    println!("Weights: {} bytes (M={} x K={})", weights.len(), m, k);
    println!("Input: {} elements (N={} x K={})", input_f32.len(), n, k);
    println!("Output: {} elements (M={} x N={})\n", output_count, m, n);

    // Create buffers
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

    // Kernel arguments (same for both kernels)
    #[repr(C)]
    struct MulMMArgs {
        ne00: i32, // K
        ne02: i32, // batch (1)
        nb01: u64, // row stride for A
        nb02: u64, // batch stride for A (0)
        nb03: u64, // batch3 stride for A (0)
        ne12: i32, // batch for B (1)
        nb10: u64, // element stride for B (4)
        nb11: u64, // row stride for B
        nb12: u64, // batch stride for B (0)
        nb13: u64, // batch3 stride for B (0)
        ne0: i32,  // M
        ne1: i32,  // N
        r2: i16,   // broadcast ratio (1)
        r3: i16,   // broadcast ratio (1)
    }

    let args = MulMMArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: (num_blocks * block_bytes) as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 4, // f32 element size
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    println!(
        "Args: ne00=K={}, nb01={}, nb11={}",
        args.ne00, args.nb01, args.nb11
    );

    // Function constants (700-705)
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

    // Test self-written kernel
    println!("\n=== Testing Self-written Kernel ===");
    let kernel_name = "kernel_mul_mm_q4_0_f32";
    let self_function =
        match self_lib.get_function(kernel_name, Some(cv.clone())) {
            Ok(f) => {
                println!("✓ Found kernel: {}", kernel_name);
                f
            }
            Err(e) => {
                println!("✗ Kernel not found: {}", e);
                println!("\nAvailable functions:");
                for name in self_lib.function_names() {
                    println!("  - {}", name);
                }
                return;
            }
        };

    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_function)
        .expect("Failed to create pipeline");

    // Run self-written kernel
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

    // Threadgroup size: 64 x 2 = 128 threads (SIMD groups)
    let threadgroup_size = MTLSize::new(64, 2, 1);
    let threadgroups =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
    encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    println!("✓ Self-written kernel executed");

    // Test llama.cpp kernel
    println!("\n=== Testing llama.cpp Kernel ===");
    let llama_function = match llama_lib.get_function(kernel_name, Some(cv)) {
        Ok(f) => {
            println!("✓ Found kernel: {}", kernel_name);
            f
        }
        Err(e) => {
            println!("✗ Kernel not found: {}", e);
            println!("\nAvailable functions:");
            for name in llama_lib.function_names() {
                println!("  - {}", name);
            }
            return;
        }
    };

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_function)
        .expect("Failed to create pipeline");

    // Run llama.cpp kernel
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

    println!("✓ llama.cpp kernel executed");

    // Compare results
    println!("\n=== Comparing Results ===");
    let self_ptr = output_self.contents() as *const f32;
    let llama_ptr = output_llama.contents() as *const f32;

    let self_results =
        unsafe { std::slice::from_raw_parts(self_ptr, output_count) };
    let llama_results =
        unsafe { std::slice::from_raw_parts(llama_ptr, output_count) };

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut num_nan_self = 0;
    let mut num_nan_llama = 0;

    for (i, (&s, &l)) in
        self_results.iter().zip(llama_results.iter()).enumerate()
    {
        if s.is_nan() {
            num_nan_self += 1;
        }
        if l.is_nan() {
            num_nan_llama += 1;
        }

        let diff = (s - l).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    println!("Self-written NaN count: {}/{}", num_nan_self, output_count);
    println!("llama.cpp NaN count: {}/{}", num_nan_llama, output_count);
    println!("Max difference: {} at index {}", max_diff, max_diff_idx);

    // Print first few results
    println!("\nFirst 10 results:");
    println!("Index  Self      llama.cpp  Diff");
    for i in 0..10.min(output_count) {
        let diff = (self_results[i] - llama_results[i]).abs();
        println!(
            "{:5}  {:9.4}  {:9.4}  {:9.4}",
            i, self_results[i], llama_results[i], diff
        );
    }

    // Summary
    println!("\n=== Summary ===");
    if num_nan_self > 0 {
        println!(
            "✗ FAIL: Self-written kernel produced {} NaN values",
            num_nan_self
        );
    } else if num_nan_llama > 0 {
        println!(
            "✗ FAIL: llama.cpp kernel produced {} NaN values",
            num_nan_llama
        );
    } else if max_diff > 1e-3 {
        println!("✗ FAIL: Max difference {} exceeds threshold", max_diff);
    } else {
        println!("✓ PASS: Results match within tolerance");
    }

    // CPU reference calculation
    println!("\n=== CPU Reference ===");
    // Q4_0 dequant: nibble n → (n - 8) * scale
    // nibble 0→-8, 1→-7, ..., 7→-1, 8→0, 9→1, ..., 15→7
    let kvalues: [f32; 16] = [
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
    ];

    // Compute expected output[0] on CPU
    let mut cpu_result = 0.0f32;
    for row_block in 0..num_blocks {
        let weight_offset = row_block * block_bytes;
        // Scale is at offset 0-1
        let scale = 2.0f32; // We set scale = 2.0 (f16 0x4000)
        // Weights are at offset 2-17
        for byte_idx in 0..16 {
            let byte = weights[weight_offset + 2 + byte_idx];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            cpu_result += scale * kvalues[lo] * 1.0; // input is all 1.0
            cpu_result += scale * kvalues[hi] * 1.0;
        }
    }
    println!("CPU expected result for output[0]: {}", cpu_result);
    println!("GPU self result for output[0]: {}", self_results[0]);
    println!("GPU llama result for output[0]: {}", llama_results[0]);
    let cpu_diff = (self_results[0] - cpu_result).abs();
    if cpu_diff > 1e-3 {
        println!("✗ FAIL: GPU result differs from CPU by {}", cpu_diff);
    } else {
        println!("✓ PASS: GPU result matches CPU calculation");
    }
}
