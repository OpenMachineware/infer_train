// Test the mul_mm_llama_combined.metal library
use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use std::ffi::c_void;

fn main() {
    println!("=== Testing mul_mm_llama_combined.metal ===\n");

    let device = Device::system_default().expect("No Metal device");

    // Load llama.cpp combined library (simple kernel template)
    let llama_combined =
        include_str!("../src/ops/metal/mul_mm_llama_combined.metal");
    let compile_opts = metal::CompileOptions::new();
    let llama_lib = device
        .new_library_with_source(llama_combined, &compile_opts)
        .expect("Failed to compile llama combined library");

    // Load self-written library
    let ggml_common = include_str!("../src/ops/metal/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_IMPL_METAL\n{}", ggml_common);
    let standalone_source =
        include_str!("../src/ops/metal/mul_mm_standalone.metal");
    let standalone_no_include = standalone_source
        .lines()
        .filter(|line| !line.contains("#include \"iq_grid_tables.h\""))
        .collect::<Vec<_>>()
        .join("\n");
    let my_combined =
        format!("{}\n{}", ggml_common_metal, standalone_no_include);
    let my_lib = device
        .new_library_with_source(&my_combined, &compile_opts)
        .expect("Failed to compile self-written library");

    // Test dimensions
    let m = 64usize;
    let k = 256usize;
    let n = 32usize;

    // Generate weights
    let nb = k / 32;
    let weights: Vec<BlockIQ4NL> = (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                qs[i] = 0x88;
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect();

    // Generate input
    let input_f16: Vec<u16> = vec![f16::from_f32(1.0).to_bits(); k * n];

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
        nb01: 18,
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

    let queue = device.new_command_queue();

    // Create buffers
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

    // Test llama.cpp simple kernel
    println!("Testing llama.cpp simple kernel (mul_mm_llama_combined.metal):");
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_iq4_nl_f32", None).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    // Different dispatch for simple kernel: one threadgroup per row
    let llama_tg_count = MTLSize::new(n as u64, m as u64, 1);
    let llama_tg_size = MTLSize::new(32, 1, 1);

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
        enc.dispatch_thread_groups(llama_tg_count, llama_tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let ptr_llama = output_llama.contents() as *const f32;
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };
    println!(
        "  Output (first 10): {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
        slice_llama[0],
        slice_llama[1],
        slice_llama[2],
        slice_llama[3],
        slice_llama[4],
        slice_llama[5],
        slice_llama[6],
        slice_llama[7],
        slice_llama[8],
        slice_llama[9]
    );
    let non_zero_llama = slice_llama.iter().filter(|&&x| x != 0.0).count();
    println!("  Non-zero: {}", non_zero_llama);

    // Test self-written advanced kernel
    println!(
        "\nTesting self-written advanced kernel (mul_mm_standalone.metal):"
    );

    // Function constants for advanced kernel
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

    let my_fn =
        my_lib.get_function("kernel_mul_mm_iq4_nl_f32", Some(cv)).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    // Advanced kernel dispatch: 64x32 tiles
    let my_tg_count =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
    let my_tg_size = MTLSize::new(32, 4, 1);

    let output_my = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
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
        enc.dispatch_thread_groups(my_tg_count, my_tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let ptr_my = output_my.contents() as *const f32;
    let slice_my = unsafe { std::slice::from_raw_parts(ptr_my, m * n) };
    println!(
        "  Output (first 10): {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
        slice_my[0],
        slice_my[1],
        slice_my[2],
        slice_my[3],
        slice_my[4],
        slice_my[5],
        slice_my[6],
        slice_my[7],
        slice_my[8],
        slice_my[9]
    );
    let non_zero_my = slice_my.iter().filter(|&&x| x != 0.0).count();
    println!("  Non-zero: {}", non_zero_my);
}
