// Verify correctness: compare kernel outputs
use half::f16;
use infer_train::quant::types::BlockIQ4NL;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

fn gen_iq4_nl(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

#[repr(C)]
struct GemmArgs {
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

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let m = 64usize;
    let k = 128usize;
    let n = 32usize;
    let seed = 42u64; // Small size for verification

    let gemm_source = include_str!("../../shaders/gemm.metal");
    let ggml_common = include_str!("../../shaders/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../../shaders/iq_grid_tables.h");
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
    let my_lib = device
        .new_library_with_source(&combined, &metal::CompileOptions::new())
        .unwrap();
    let llama_lib = device
        .new_library_with_file(std::path::Path::new(
            "shaders/mul_mm_template.metallib",
        ))
        .unwrap();

    let weights = gen_iq4_nl(m, k, seed);
    let input_f16 = gen_f16_input(k, n, seed);

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 18,
        nb01: (k / 32 * 18) as u64,
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

    let cv = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
    unsafe {
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
    }

    let my_pipeline = device
        .new_compute_pipeline_state_with_function(
            &my_lib
                .get_function("kernel_gemm_iq4_nl_f32", Some(cv.clone()))
                .unwrap(),
        )
        .unwrap();
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib
                .get_function("kernel_mul_mm_iq4_nl_f32", Some(cv))
                .unwrap(),
        )
        .unwrap();

    let queue = device.new_command_queue();
    let output_my = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Run self
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
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run llama
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

    // Compare results
    let my_data = unsafe {
        std::slice::from_raw_parts(output_my.contents() as *const f32, m * n)
    };
    let llama_data = unsafe {
        std::slice::from_raw_parts(output_llama.contents() as *const f32, m * n)
    };

    let mut max_diff = 0.0f32;
    let mut diff_count = 0usize;
    for i in 0..m * n {
        let diff = (my_data[i] - llama_data[i]).abs();
        if diff > 0.001 {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    println!("Results comparison:");
    println!("  Total elements: {}", m * n);
    println!("  Mismatches (>0.001): {}", diff_count);
    println!("  Max difference: {:.6}", max_diff);
    println!(
        "  Sample [0]: self={:.6}, llama={:.6}",
        my_data[0], llama_data[0]
    );
    println!(
        "  Sample [1]: self={:.6}, llama={:.6}",
        my_data[1], llama_data[1]
    );

    if diff_count == 0 {
        println!("\n✓ Results MATCH - both kernels compute correctly");
    } else {
        println!("\n✗ Results DIFFER - configuration error!");
    }
}
