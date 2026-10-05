// Simple test for IQ GEMM - test one kernel at a time
// Test with simple known values instead of random data

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use std::ffi::c_void;

fn main() {
    println!("=== Simple IQ GEMM Test ===\n");

    let device = Device::system_default().expect("No Metal device");

    // Test dimensions
    let m = 64usize; // Small for debugging
    let k = 256usize; // One block
    let n = 32usize; // True GEMM

    println!("M={}, K={}, N={}", m, k, n);

    // Load llama.cpp library
    let llama_lib_path =
        std::path::Path::new("shaders/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp library");

    // Load self-written library
    let ggml_common = include_str!("../../shaders/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_IMPL_METAL\n{}", ggml_common);
    let standalone_source =
        include_str!("../../shaders/mul_mm_standalone.metal");
    let standalone_no_include = standalone_source
        .lines()
        .filter(|line| !line.contains("#include \"iq_grid_tables.h\""))
        .collect::<Vec<_>>()
        .join("\n");
    let combined = format!("{}\n{}", ggml_common_metal, standalone_no_include);
    let compile_opts = metal::CompileOptions::new();
    let my_lib = device
        .new_library_with_source(&combined, &compile_opts)
        .expect("Failed to compile library");

    // Test IQ4_NL (simplest IQ format - just 4-bit lookup)
    println!("\n=== Testing IQ4_NL (no grid lookup, just kvalues) ===");

    // Generate simple weights: all zeros for qs, scale = 1.0
    let nb = k / 32;
    let weights: Vec<BlockIQ4NL> = (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            // Set some non-zero values
            for i in 0..16 {
                qs[i] = 0x88; // All bits set to 8 -> indices 8 and 8
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect();

    // Generate simple input: all ones
    let input_f16: Vec<u16> = vec![f16::from_f32(1.0).to_bits(); k * n];

    // Args for IQ4_NL (32-element block, 18 bytes)
    #[repr(C)]
    struct GemmArgs {
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

    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 18, // 18 bytes per block
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    // Function constants
    const FC_MUL_MM: usize = 700;
    let cv = metal::FunctionConstantValues::new();
    let one_val: i16 = 1;
    let false_val: bool = false;
    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        FC_MUL_MM as u64,
    );
    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        (FC_MUL_MM + 1) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 2) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 3) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 4) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 5) as u64,
    );

    let queue = device.new_command_queue();

    // Test self-written kernel
    let my_fn = my_lib
        .get_function("kernel_mul_mm_iq4_nl_f32", Some(cv.clone()))
        .unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    let weights_bytes: &[u8] = bytemuck::cast_slice(&weights);
    let weights_buffer = device.new_buffer_with_data(
        weights_bytes.as_ptr() as *const c_void,
        weights_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_my = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);

    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_my), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Test llama.cpp kernel
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_iq4_nl_f32", Some(cv)).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare outputs
    let ptr_my = output_my.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_my = unsafe { std::slice::from_raw_parts(ptr_my, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    println!("Self-written kernel output (first 10 values):");
    for i in 0..10.min(slice_my.len()) {
        print!("  {:.4}", slice_my[i]);
    }
    println!();

    println!("llama.cpp kernel output (first 10 values):");
    for i in 0..10.min(slice_llama.len()) {
        print!("  {:.4}", slice_llama[i]);
    }
    println!();

    let non_zero_my = slice_my.iter().filter(|&&x| x != 0.0).count();
    let non_zero_llama = slice_llama.iter().filter(|&&x| x != 0.0).count();
    println!("Non-zero: self={}, llama={}", non_zero_my, non_zero_llama);

    let sum_my: f64 = slice_my.iter().map(|&x| x as f64).sum();
    let sum_llama: f64 = slice_llama.iter().map(|&x| x as f64).sum();
    println!("Sum: self={:.2}, llama={:.2}", sum_my, sum_llama);
}
