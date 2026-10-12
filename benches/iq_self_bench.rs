// Self-written IQ kernel performance benchmark
// Tests only self-written kernels, no llama.cpp comparison

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

fn main() {
    println!("=== Self-written IQ Kernel Performance ===\n");

    let device = Device::system_default().expect("No Metal device");

    // Compile library
    let ggml_common = include_str!("../src/ops/metal/blocks.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../src/ops/metal/iq_grid_tables.h");
    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");

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

    let combined = format!(
        "{}\n{}\n{}\n{}",
        ggml_common_metal, iq_extra_tables, iq_grid_tables, gemm_source
    );

    let compile_opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(&combined, &compile_opts)
        .expect("Failed to compile GEMM library");

    // Test sizes
    let m = 4096;
    let k = 4096;
    let seed = 42u64;

    // Args structure
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

    // Function constants helper
    const FC_MUL_MV: usize = 600;
    let nsg_val: i16 = 4;
    let one_val: i16 = 1;

    let create_cv = || {
        let cv = metal::FunctionConstantValues::new();
        cv.set_constant_value_at_index(
            &nsg_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            FC_MUL_MV as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MV + 2) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MV + 3) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MV + 4) as u64,
        );
        cv
    };

    let queue = device.new_command_queue();

    // Test each IQ format
    let kernels: [(&str, &str, i32); 9] = [
        ("IQ1_S", "kernel_mul_mm_iq1_s_f32", 56), // Block size 56 bytes
        ("IQ1_M", "kernel_mul_mm_iq1_m_f32", 56),
        ("IQ2_XXS", "kernel_mul_mm_iq2_xxs_f32", 66),
        ("IQ2_XS", "kernel_mul_mm_iq2_xs_f32", 74),
        ("IQ2_S", "kernel_mul_mm_iq2_s_f32", 82),
        ("IQ3_XXS", "kernel_mul_mm_iq3_xxs_f32", 98),
        ("IQ3_S", "kernel_mul_mm_iq3_s_f32", 110),
        ("IQ4_NL", "kernel_mul_mm_iq4_nl_f32", 18), // Special: 32 elements/block
        ("IQ4_XS", "kernel_mul_mm_iq4_xs_f32", 136),
    ];

    println!("Format      Time (µs)   Output (sum)   Status");
    println!("{}", "-".repeat(50));

    for (name, kernel_name, block_size) in kernels.iter() {
        let nb = if *name == "IQ4_NL" { k / 32 } else { k / 256 };
        let block_size_usize = *block_size as usize;

        // Generate weights
        let mut rng = StdRng::seed_from_u64(seed);
        let weights: Vec<u8> =
            (0..m * nb * block_size_usize).map(|_| rng.r#gen::<u8>()).collect();

        // Generate input
        let input_f16: Vec<u16> = (0..k)
            .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
            .collect();

        let args = GemmArgsMv {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: *block_size as u64,
            nb01: (nb * block_size_usize) as u64,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 1,
            r2: 1,
            r3: 1,
        };

        let fn_name = lib.get_function(kernel_name, Some(create_cv()));
        if fn_name.is_err() {
            println!("{:<12} {:>10} {:>14}   SKIP (not found)", name, "-", "-");
            continue;
        }

        let pipeline = device
            .new_compute_pipeline_state_with_function(&fn_name.unwrap())
            .unwrap();

        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            (m * nb * *block_size as usize) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input_f16.as_ptr() as *const c_void,
            (k * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Dispatch
        let nsg = 4;
        let nr0 = 1;
        let rows_per_tg = nsg * nr0;
        let tg_size = MTLSize::new(32, nsg as u64, 1);
        let tg_count =
            MTLSize::new(1, ((m + rows_per_tg - 1) / rows_per_tg) as u64, 1);

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgsMv>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgsMv>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let elapsed = start.elapsed().as_micros() as f64 / iterations as f64;

        // Read output
        let ptr = output.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, m) };
        let non_zero = slice.iter().filter(|&&x| x != 0.0).count();
        let sum: f64 = slice.iter().map(|&x| x as f64).sum();

        let status = if non_zero == m { "OK" } else { "WARN" };
        println!("{:<12} {:>10.2} {:>14.2}   {}", name, elapsed, sum, status);
    }

    println!("\nDone.");
}
