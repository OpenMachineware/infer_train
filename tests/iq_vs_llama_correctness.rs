// Compare correctness: self-written gemm.metal vs llama.cpp mul_mm_template.metal
// Uses minimal data to verify correctness without risking system stability

use half::f16;
use metal::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

fn main() {
    println!("=== IQ Kernel Correctness: Self vs llama.cpp ===");
    println!("Using minimal data (M=4, K=256, N=1) for safety\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Compile self-written shader
    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/ggml-common.h");
    let ggml_common_metal = format!(
        "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n{}",
        ggml_common
    );
    let my_combined = format!("{}\n{}", ggml_common_metal, gemm_source);

    println!("Compiling self-written shader...");
    let my_lib = device
        .new_library_with_source(&my_combined, &metal::CompileOptions::new())
        .expect("Failed to compile self-written library");
    println!("✓ Self-written shader compiled\n");

    // Load llama.cpp library
    println!("Loading llama.cpp library...");
    let llama_lib = device
        .new_library_with_file(std::path::Path::new(
            "src/ops/metal/mul_mm_template.metallib",
        ))
        .expect("Failed to load llama.cpp library");
    println!("✓ llama.cpp library loaded\n");

    // Test parameters
    let m = 4;
    let k = 256;
    let n = 1;
    let num_blocks = k / 32; // QK4_NL = 32

    // Generate random weights
    let mut rng = StdRng::seed_from_u64(42);
    let mut weights = Vec::with_capacity(m * num_blocks * 18);
    for _ in 0..m * num_blocks {
        weights.push(rng.r#gen::<u8>()); // d low byte
        weights.push(rng.r#gen::<u8>()); // d high byte
        for _ in 0..16 {
            weights.push(rng.r#gen::<u8>()); // packed nibbles
        }
    }

    // Generate input (f32)
    let input_f32: Vec<f32> = (0..k).map(|_| rng.r#gen::<f32>()).collect();

    println!(
        "Weights: {} bytes ({} blocks x {} rows)",
        weights.len(),
        num_blocks,
        m
    );
    println!("Input: {} elements (f32)", k);
    println!("Output: {} f32\n", m);

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

    // Kernel arguments
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

    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 18,
        nb01: 18 * num_blocks as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        nr0: 2, // N_R0_IQ4_NL
        r2: 1,
        r3: 1,
    };

    // Function constants (FC_MUL_MV range for self, FC_MUL_MM range for llama.cpp)
    let cv_self = metal::FunctionConstantValues::new();
    let nsg: i16 = 2;
    let ne12: i16 = 1;
    let r2: i16 = 1;
    let r3: i16 = 1;
    unsafe {
        cv_self.set_constant_value_at_index(
            &nsg as *const _ as *const c_void,
            metal::MTLDataType::Short,
            600,
        );
        cv_self.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            602,
        );
        cv_self.set_constant_value_at_index(
            &r2 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            603,
        );
        cv_self.set_constant_value_at_index(
            &r3 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            604,
        );
    }

    let cv_llama = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    unsafe {
        cv_llama.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            700,
        );
        cv_llama.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            701,
        );
        cv_llama.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            702,
        );
        cv_llama.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            703,
        );
        cv_llama.set_constant_value_at_index(
            &r2 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            704,
        );
        cv_llama.set_constant_value_at_index(
            &r3 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            705,
        );
    }

    // Create pipelines
    println!("Creating pipelines...");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(
            &my_lib
                .get_function("kernel_mul_mm_iq4_nl_f32", Some(cv_self))
                .unwrap(),
        )
        .unwrap();
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib
                .get_function("kernel_mul_mm_iq4_nl_f32", Some(cv_llama))
                .unwrap(),
        )
        .unwrap();
    println!("✓ Pipelines created\n");

    // Run both
    let output_self = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let output_llama = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    for (name, pipeline, output) in [
        ("Self", &my_pipeline, &output_self),
        ("llama.cpp", &llama_pipeline, &output_llama),
    ] {
        println!("Running {} kernel...", name);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(output), 0);
        enc.set_threadgroup_memory_length(0, 8192);

        let tg_size = MTLSize::new(32, 4, 1);
        let tg_count = MTLSize::new(1, 1, 1);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        println!("✓ {} kernel done", name);
    }
    println!();

    // Compare results
    let self_data = unsafe {
        std::slice::from_raw_parts(output_self.contents() as *const f32, m)
    };
    let llama_data = unsafe {
        std::slice::from_raw_parts(output_llama.contents() as *const f32, m)
    };

    println!("=== Results ===");
    println!("Self[0..{}]: {:?}", m, &self_data[0..m.min(4)]);
    println!("Llama[0..{}]: {:?}", m, &llama_data[0..m.min(4)]);

    let mut max_diff = 0.0f32;
    for i in 0..m {
        let diff = (self_data[i] - llama_data[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("\nMax difference: {:.6}", max_diff);

    if max_diff < 0.001 {
        println!("\n✓ PASS: Results match llama.cpp within tolerance");
    } else {
        println!("\n✗ FAIL: Results differ from llama.cpp");
    }
}
