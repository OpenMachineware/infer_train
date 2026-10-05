// Test all quantization formats using quantize functions
use half::f16;
use metal::{Device, MTLSize};
use rand::Rng;
use std::ffi::c_void;

use infer_train::quant::types::*;
use infer_train::quant::weights::*;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GemmArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: i64,
    pub nb01: i64,
    pub nb02: i64,
    pub nb03: i64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: i64,
    pub nb11: i64,
    pub nb12: i64,
    pub nb13: i64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: i64,
    pub nb1: i64,
    pub nb2: i64,
    pub nb3: i64,
    pub r2: i32,
    pub r3: i32,
}

fn create_function_constants() -> metal::FunctionConstantValues {
    let cv = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;

    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        700,
    );
    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        701,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        702,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        703,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        704,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        705,
    );
    cv.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        706,
    );
    cv
}

fn test_format<B: bytemuck::Pod + bytemuck::Zeroable + Copy>(
    device: &Device,
    queue: &metal::CommandQueue,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    kernel_name: &str,
    weights: &[B],
    block_size: usize,
    input_f16: &[f16],
    m: usize,
    k: usize,
    n: usize,
    format_name: &str,
) {
    let cv = create_function_constants();

    let my_pipeline = device
        .new_compute_pipeline_state_with_function(
            &my_lib.get_function(kernel_name, Some(cv.clone())).unwrap(),
        )
        .unwrap();
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib.get_function(kernel_name, Some(cv)).unwrap(),
        )
        .unwrap();

    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        ne03: 1,
        nb00: block_size as i64,
        nb01: (k / 256 * block_size) as i64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        ne13: 1,
        nb10: 2,
        nb11: (k * 2) as i64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        ne2: 1,
        ne3: 1,
        nb0: 4,
        nb1: (m * 4) as i64,
        nb2: 0,
        nb3: 0,
        r2: 1,
        r3: 1,
    };

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<B>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_my = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((n + 63) / 64) as u64, ((m + 63) / 64) as u64, 1);

    // Run my kernel
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
        enc.set_buffer(3, Some(&output_my), 0);
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

    // Compare
    let my_ptr = output_my.contents() as *const f32;
    let llama_ptr = output_llama.contents() as *const f32;
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for i in 0..(m * n) {
        let my_val = unsafe { *my_ptr.add(i) };
        let llama_val = unsafe { *llama_ptr.add(i) };
        let diff = (my_val - llama_val).abs();
        if diff > 0.001 {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    if diff_count == 0 {
        println!("{:<12} ✓ PASS", format_name);
    } else {
        println!(
            "{:<12} ✗ FAIL ({}/{} diff, max={:.4})",
            format_name,
            diff_count,
            m * n,
            max_diff
        );
    }
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let my_lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .unwrap();
    let llama_lib = device
        .new_library_with_file("shaders/mul_mm_template.metallib")
        .unwrap();

    // Test parameters
    let m = 64;
    let k = 256;
    let n = 64;

    // Generate random FP32 test data
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(42);

    let weights_f32: Vec<f32> =
        (0..(m * k)).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let input_f32: Vec<f32> =
        (0..(k * n)).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let input_f16: Vec<f16> =
        input_f32.iter().map(|&f| f16::from_f32(f)).collect();

    println!("=== GEMM Correctness Test (M={}, K={}, N={}) ===\n", m, k, n);
    println!("{:<12} Status", "Format");
    println!("{}", "-".repeat(24));

    // Standard Q formats
    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q4_0_f16",
        &quantize_q4_0(&weights_f32),
        std::mem::size_of::<BlockQ4_0>(),
        &input_f16,
        m,
        k,
        n,
        "Q4_0",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q8_0_f16",
        &quantize_q8_0(&weights_f32),
        std::mem::size_of::<BlockQ8_0>(),
        &input_f16,
        m,
        k,
        n,
        "Q8_0",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q4_1_f16",
        &quantize_q4_1(&weights_f32),
        std::mem::size_of::<BlockQ4_1>(),
        &input_f16,
        m,
        k,
        n,
        "Q4_1",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q5_0_f16",
        &quantize_q5_0(&weights_f32),
        std::mem::size_of::<BlockQ5_0>(),
        &input_f16,
        m,
        k,
        n,
        "Q5_0",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q5_1_f16",
        &quantize_q5_1(&weights_f32),
        std::mem::size_of::<BlockQ5_1>(),
        &input_f16,
        m,
        k,
        n,
        "Q5_1",
    );

    // K-quant formats
    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q2_K_f16",
        &quantize_q2_k(&weights_f32),
        std::mem::size_of::<BlockQ2K>(),
        &input_f16,
        m,
        k,
        n,
        "Q2_K",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q3_K_f16",
        &quantize_q3_k(&weights_f32),
        std::mem::size_of::<BlockQ3K>(),
        &input_f16,
        m,
        k,
        n,
        "Q3_K",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q4_K_f16",
        &quantize_q4_k(&weights_f32),
        std::mem::size_of::<BlockQ4K>(),
        &input_f16,
        m,
        k,
        n,
        "Q4_K",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q5_K_f16",
        &quantize_q5_k(&weights_f32),
        std::mem::size_of::<BlockQ5K>(),
        &input_f16,
        m,
        k,
        n,
        "Q5_K",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_q6_K_f16",
        &quantize_q6_k(&weights_f32),
        std::mem::size_of::<BlockQ6K>(),
        &input_f16,
        m,
        k,
        n,
        "Q6_K",
    );

    // IQ formats
    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq4_nl_f16",
        &quantize_iq4_nl(&weights_f32),
        std::mem::size_of::<BlockIQ4NL>(),
        &input_f16,
        m,
        k,
        n,
        "IQ4_NL",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq4_xs_f16",
        &quantize_iq4_xs(&weights_f32),
        std::mem::size_of::<BlockIQ4XS>(),
        &input_f16,
        m,
        k,
        n,
        "IQ4_XS",
    );

    // For IQ ultra-low-bit formats, use test quantizers
    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq1_s_f16",
        &quantize_iq1_s_test(&weights_f32),
        std::mem::size_of::<BlockIQ1S>(),
        &input_f16,
        m,
        k,
        n,
        "IQ1_S",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq2_xxs_f16",
        &quantize_iq2_xxs_test(&weights_f32),
        std::mem::size_of::<BlockIQ2XXS>(),
        &input_f16,
        m,
        k,
        n,
        "IQ2_XXS",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq2_xs_f16",
        &quantize_iq2_xs_test(&weights_f32),
        std::mem::size_of::<BlockIQ2XS>(),
        &input_f16,
        m,
        k,
        n,
        "IQ2_XS",
    );

    test_format(
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        "kernel_mul_mm_iq3_xxs_f16",
        &quantize_iq3_xxs_test(&weights_f32),
        std::mem::size_of::<BlockIQ3XXS>(),
        &input_f16,
        m,
        k,
        n,
        "IQ3_XXS",
    );

    println!("\nDone.");
}
