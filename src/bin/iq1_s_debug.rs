// Debug test for IQ1_S GEMM
// Compare individual values between self-written and llama.cpp kernels

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ffi::c_void;

const QK_K: usize = 256;

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

fn gen_iq1_s_known(m: usize, k: usize) -> Vec<BlockIQ1S> {
    // Generate blocks with known pattern for debugging
    let nb = k / QK_K;
    (0..m * nb)
        .map(|block_idx| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            // Set simple pattern: all qs = 0x55 (alternating bits)
            for j in 0..32 {
                qs[j] = 0x55;
            }
            // Set qh with known pattern
            for j in 0..8 {
                qh[j] = 0x1234;
            }
            BlockIQ1S { d: f16::from_f32(1.0).to_bits(), qs, qh }
        })
        .collect()
}

fn gen_iq1_s_zeros(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1S> {
    // Generate blocks with random values for testing
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u16>();
            }
            BlockIQ1S { d: f16::from_f32(1.0).to_bits(), qs, qh }
        })
        .collect()
}

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

fn main() {
    println!("=== IQ1_S Debug Test ===\n");

    let device = Device::system_default().expect("No Metal device");

    // Small test size
    let m = 1usize;
    let k = 256usize;
    let n = 1usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}\n", m, k, n);

    // Load libraries
    let my_lib_path =
        std::path::Path::new("shaders/mul_mm_standalone.metallib");
    let my_lib = device
        .new_library_with_file(my_lib_path)
        .expect("Failed to load standalone GEMM library");

    let llama_lib_path =
        std::path::Path::new("shaders/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp template library");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    // Use known values for debugging
    let weights = gen_iq1_s_known(m, k);

    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 50,
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

    let my_fn = my_lib
        .get_function("kernel_mul_mm_iq1_s_f16", Some(cv.clone()))
        .unwrap();
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_iq1_s_f16", Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    // Print pipeline info
    println!("Pipeline thread execution width:");
    println!("  self: {}", my_pipeline.thread_execution_width());
    println!("  llama: {}", llama_pipeline.thread_execution_width());
    println!();

    let queue = device.new_command_queue();

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * 50) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_self = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Dispatch
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(1, 1, 1);

    // Run self kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Print first few values from weights buffer
    println!("First 16 bytes of weights:");
    let weights_ptr = weights_buf.contents() as *const u8;
    let weights_slice = unsafe { std::slice::from_raw_parts(weights_ptr, 16) };
    for (i, b) in weights_slice.iter().enumerate() {
        print!("{:02x} ", b);
        if (i + 1) % 8 == 0 {
            println!();
        }
    }
    println!();

    // Compare outputs
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;

    println!("\nOutputs:");
    println!("  self:  {}", unsafe { *ptr_self });
    println!("  llama: {}", unsafe { *ptr_llama });

    let diff = unsafe { (*ptr_self - *ptr_llama).abs() };
    println!("  diff:  {}", diff);

    if diff < 1e-6 {
        println!("\n✅ PASSED");
    } else {
        println!("\n❌ FAILED");
    }
}
