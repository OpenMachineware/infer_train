// Element-by-element comparison for IQ4_NL kernel

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

fn main() {
    println!("=== IQ4_NL Correctness Debug ===\n");

    let device = Device::system_default().expect("No Metal device");

    // Compile libraries
    let ggml_common = include_str!("../../shaders/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../../shaders/iq_grid_tables.h");
    let gemm_source = include_str!("../../shaders/gemm.metal");

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

    let llama_lib_path =
        std::path::Path::new("shaders/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp library");

    // Test with small sizes for detailed comparison
    let m = 8; // 8 rows
    let k = 256; // 1 super-block per row
    let n = 1;
    let seed = 42u64;
    let nb = k / 32;

    // Generate weights - all with scale=1.0 for simplicity
    let mut rng = StdRng::seed_from_u64(seed);
    let weights: Vec<BlockIQ4NL> = (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect();

    // Generate input
    let input_f16: Vec<u16> = (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect();

    println!("Test: M={}, K={}, N={}", m, k, n);
    println!(
        "Weights: {} blocks x {} bytes = {} bytes",
        m * nb,
        18,
        m * nb * 18
    );
    println!("Input: {} x {} f16 = {} bytes", k, n, k * 2);

    // Setup args for self-written kernel
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

    // Args for llama.cpp template
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

    // Function constants for self-written kernel
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

    // Function constants for llama.cpp kernel
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
        (m * nb * 18) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (k * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_self = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Run self-written kernel
    // Dispatch: nr0=2, NSG=4, rows_per_tg = 8
    let rows_per_tg = 8;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(1, ((m + rows_per_tg - 1) / rows_per_tg) as u64, 1);

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
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 64);
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
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare outputs
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m) };

    println!("\n=== Output Comparison ===");
    println!(
        "{:<8} {:>15} {:>15} {:>12}",
        "Row", "Self-written", "llama.cpp", "Diff"
    );
    println!("{}", "-".repeat(55));

    let mut total_diff = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_diff_row = 0;

    for i in 0..m {
        let diff = (slice_self[i] - slice_llama[i]).abs();
        total_diff += diff as f64;
        if diff > max_diff {
            max_diff = diff;
            max_diff_row = i;
        }

        let marker = if diff < 0.001 { "" } else { " *" };
        println!(
            "{:<8} {:>15.4} {:>15.4} {:>12.4}{}",
            i, slice_self[i], slice_llama[i], diff, marker
        );
    }

    println!(
        "\nSum (self):      {:.4}",
        slice_self.iter().map(|&x| x as f64).sum::<f64>()
    );
    println!(
        "Sum (llama.cpp): {:.4}",
        slice_llama.iter().map(|&x| x as f64).sum::<f64>()
    );
    println!("Max diff:        {:.4} (row {})", max_diff, max_diff_row);
    println!("Avg diff:        {:.4}", total_diff / m as f64);

    // Check first row in detail
    println!("\n=== First Row Detailed Analysis ===");

    // Compute expected value on CPU
    let kvalues: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0,
        25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
    ];
    let input_f32: Vec<f32> =
        input_f16.iter().map(|&bits| f16::from_bits(bits).to_f32()).collect();

    let mut cpu_sum = 0.0f32;
    for b in 0..nb {
        let scale = f16::from_bits(weights[b].d).to_f32();
        for i in 0..16 {
            let lo_nibble = (weights[b].qs[i] & 0x0F) as usize;
            let hi_nibble = ((weights[b].qs[i] >> 4) & 0x0F) as usize;
            let idx = b * 32 + 2 * i;
            cpu_sum += scale * input_f32[idx] * kvalues[lo_nibble];
            cpu_sum += scale * input_f32[idx + 1] * kvalues[hi_nibble];
        }
    }

    println!("CPU expected:    {:.4}", cpu_sum);
    println!("GPU self:        {:.4}", slice_self[0]);
    println!("GPU llama.cpp:   {:.4}", slice_llama[0]);
    println!(
        "Diff (CPU-GPU):  self={:.4}, llama={:.4}",
        (cpu_sum - slice_self[0]).abs(),
        (cpu_sum - slice_llama[0]).abs()
    );

    println!("\nDone.");
}
