// Debug IQ kernels with increasing sizes to find crash point

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ffi::c_void;

fn test_iq4_nl(
    device: &Device,
    my_lib: &metal::Library,
    m: usize,
    k: usize,
    seed: u64,
) -> Result<(f32, f32), String> {
    let n = 1;
    let nb = k / 32;

    // Generate weights
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

    // Setup args
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

    let my_fn = my_lib
        .get_function("kernel_gemm_iq4_nl_f32", Some(cv_my))
        .map_err(|e| format!("Failed to get function: {:?}", e))?;
    let my_pipeline =
        device
            .new_compute_pipeline_state_with_function(&my_fn)
            .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

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
    let output_buffer = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Dispatch: nr0=2, NSG=4, so rows_per_tg = 8
    let rows_per_tg = 8;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((n + 31) / 32) as u64,
        ((m + rows_per_tg - 1) / rows_per_tg) as u64,
        1,
    );

    // Run kernel
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
    enc.set_buffer(3, Some(&output_buffer), 0);
    enc.set_threadgroup_memory_length(0, 64);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Read output
    let ptr = output_buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, m) };

    let sum: f64 = slice.iter().map(|&x| x as f64).sum();
    let non_zero = slice.iter().filter(|&&x| x != 0.0).count();

    Ok((sum as f32, non_zero as f32))
}

fn main() {
    println!("=== IQ4_NL Kernel Debug ===\n");

    // Load libraries
    let device = Device::system_default().expect("No Metal device");

    let ggml_common = include_str!("../../shaders/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../../shaders/iq_grid_tables.h");
    let gemm_source = include_str!("../../shaders/gemm.metal");

    // Add missing IQ tables
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

    // Test with increasing sizes
    let seed = 42u64;
    let test_cases = [
        (1, 32),      // 1 block
        (1, 256),     // 8 blocks
        (1, 1024),    // 32 blocks
        (8, 256),     // 8 rows
        (16, 256),    // 16 rows
        (32, 256),    // 32 rows
        (64, 1024),   // 64 rows, larger K
        (128, 1024),  // 128 rows
        (256, 4096),  // 256 rows, K=4096
        (512, 4096),  // 512 rows
        (1024, 4096), // 1024 rows
        (2048, 4096), // 2048 rows
        (4096, 4096), // Full size
    ];

    for (m, k) in test_cases {
        print!("Testing M={}, K={}... ", m, k);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        match test_iq4_nl(&device, &my_lib, m, k, seed) {
            Ok((sum, non_zero)) => {
                let expected_non_zero = m as f32;
                if non_zero == expected_non_zero {
                    println!("OK (sum={:.2}, non_zero={:.0})", sum, non_zero);
                } else {
                    println!(
                        "WARN (sum={:.2}, non_zero={:.0}, expected={:.0})",
                        sum, non_zero, expected_non_zero
                    );
                }
            }
            Err(e) => {
                println!("CRASH: {}", e);
                break;
            }
        }

        // Small delay to let GPU recover
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("\nDone.");
}
