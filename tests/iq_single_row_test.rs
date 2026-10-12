// Minimal single-row IQ4_NL test
// Tests just 1 row with 1 threadgroup to debug the kernel

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

fn main() {
    println!("=== Single Row IQ4_NL Test ===");

    let device = Device::system_default().expect("No Metal device");

    // Minimal test: M=1, K=256, N=1
    // This tests 8 blocks of 32 elements each
    let m = 1usize;
    let k = 256usize;
    let n = 1usize;
    let seed = 42u64;

    println!("M={}, K={}, N={} (single block test)\n", m, k, n);

    // Load self-written gemm.metal
    let gemm_source = include_str!("../../shaders/gemm.metal");
    let ggml_common = include_str!("../../shaders/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../../shaders/iq_grid_tables.h");

    // IQ kernel needs ksigns/kmask tables
    let iq_extra_tables = r#"
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

    let my_combined = format!(
        "{}\n{}\n{}\n{}",
        ggml_common_metal, iq_extra_tables, iq_grid_tables, gemm_source
    );

    let compile_opts = metal::CompileOptions::new();
    let my_lib = device
        .new_library_with_source(&my_combined, &compile_opts)
        .expect("Failed to compile GEMM library");

    // Load pre-compiled llama.cpp library
    let llama_lib_path =
        std::path::Path::new("shaders/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp library");

    // Generate input (K=32, N=1)
    let mut rng = StdRng::seed_from_u64(seed);
    let input_f32: Vec<f32> =
        (0..k * n).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let input_f16: Vec<u16> =
        input_f32.iter().map(|&f| f16::from_f32(f).to_bits()).collect();

    // Generate IQ4_NL weights (8 blocks for K=256, 18 bytes each)
    let nb = k / 32;
    let mut weights: Vec<BlockIQ4NL> = Vec::with_capacity(nb);
    for _ in 0..nb {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = rng.r#gen::<u8>();
        }
        weights.push(BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs });
    }

    println!(
        "\nInput (first 8 f32): {:?}",
        &input_f32[..8.min(input_f32.len())]
    );
    println!(
        "Input (first 8 f16): {:04x?}",
        &input_f16[..8.min(input_f16.len())]
    );
    println!("Weight qs: {:02x?}", &weights[0].qs);
    println!("Weight d: {:04x} (f16)", weights[0].d);

    // Compute expected value on CPU using the actual f16-quantized values
    // kvalues from ggml-common.h: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    let kvalues: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0,
        25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
    ];

    // Convert f16 back to f32 for accurate CPU computation
    let input_f16_as_f32: Vec<f32> =
        input_f16.iter().map(|&bits| f16::from_bits(bits).to_f32()).collect();

    let mut expected = 0.0f32;
    for b in 0..nb {
        let scale = f16::from_bits(weights[b].d).to_f32();
        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let lo_nibble = (weights[b].qs[i] & 0x0F) as usize;
            let hi_nibble = ((weights[b].qs[i] >> 4) & 0x0F) as usize;
            let idx = b * 32 + 2 * i;
            let val_lo = input_f16_as_f32[idx] * kvalues[lo_nibble];
            let val_hi = input_f16_as_f32[idx + 1] * kvalues[hi_nibble];
            block_sum += val_lo + val_hi;
        }
        expected += scale * block_sum;
    }
    println!("CPU expected (f16 precision): {:.4}", expected);

    // Create argument structures
    #[repr(C)]
    struct GemmArgsMv {
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

    #[repr(C)]
    struct GemmArgsMm {
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

    let args_mv = GemmArgsMv {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 18,
        nb01: (nb * 18) as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let args_mm = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (nb * 18) as u64,
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
    const FC_MUL_MV: usize = 600;
    let cv_my = metal::FunctionConstantValues::new();
    let nsg_val: i16 = 4;
    let one_val: i16 = 1;
    cv_my.set_constant_value_at_index(
        &nsg_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        FC_MUL_MV as u64,
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 2) as u64,
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 3) as u64,
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 4) as u64,
    );

    const FC_MUL_MM: usize = 700;
    let cv_llama = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    cv_llama.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        FC_MUL_MM as u64,
    );
    cv_llama.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        (FC_MUL_MM + 1) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 2) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 3) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 4) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 5) as u64,
    );

    let my_fn =
        my_lib.get_function("kernel_gemm_iq4_nl_f32", Some(cv_my)).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    let llama_fn = llama_lib
        .get_function("kernel_mul_mm_iq4_nl_f32", Some(cv_llama))
        .unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let queue = device.new_command_queue();
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (nb * 18) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (k * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_a =
        device.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
    let output_b =
        device.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);

    // Dispatch: llama.cpp uses 128 threads total (32 x 4 SIMD groups)
    // But the actual thread count depends on the kernel template parameters
    // For IQ4_NL with nl=2, NK=32, NL0=2, NL1=4
    // Total threads = NL0 * nr0 * NL1 * nr1 = 2 * 1 * 4 * 1 = 8 threads needed
    // But Metal dispatches 32 threads per SIMD, so we use 32 x 4 = 128 threads
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(1, 1, 1);

    // Run self-written kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMv>() as u64,
            &args_mv as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_a), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama.cpp kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args_mm as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_b), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare
    let ptr_a = output_a.contents() as *const f32;
    let ptr_b = output_b.contents() as *const f32;
    let val_a = unsafe { *ptr_a };
    let val_b = unsafe { *ptr_b };

    println!("\nResults:");
    println!("  CPU expected (f16):  {:.4}", expected);
    println!("  Self-written:       {:.4}", val_a);
    println!("  llama.cpp:          {:.4}", val_b);
    println!("  Difference (GPU):   {:.4}", (val_a - val_b).abs());
    println!("  Difference (CPU):   {:.4}", (val_a - expected).abs());

    if (val_a - val_b).abs() < 1.0 && (val_a - expected).abs() < 10.0 {
        println!("  PASS: Values match within tolerance!");
    } else {
        println!("  FAIL: Values differ!");
    }
}
