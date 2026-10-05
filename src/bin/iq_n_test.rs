// Fixed: Match llama.cpp's ggml_metal_kargs_mul_mm structure exactly
use half::f16;
use infer_train::quant::types::BlockIQ4NL;
use metal::{Device, MTLSize};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ffi::c_void;
use std::time::Instant;

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

// Match llama.cpp's ggml_metal_kargs_mul_mm exactly!
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
    let m = 4096usize;
    let k = 4096usize;
    let seed = 42u64;

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
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    println!("N | Self(µs) | llama(µs) | Ratio");
    println!("--|----------|-----------|------");

    for n in [32usize, 64, 128, 256, 512, 1024, 2048, 4096, 8192].iter() {
        let input_f16 = gen_f16_input(k, *n, seed);
        let input_buffer = device.new_buffer_with_data(
            input_f16.as_ptr() as *const c_void,
            (input_f16.len() * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Correct args matching llama.cpp structure
        let args = GemmArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 18,
            nb01: (k / 32 * 18) as u64,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: *n as i32,
            ne12: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: *n as i32,
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

        // Warmup
        for _ in 0..3 {
            for pipeline in [&my_pipeline, &llama_pipeline].iter() {
                let output = device.new_buffer(
                    (m * n * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
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
                enc.set_buffer(3, Some(&output), 0);
                enc.set_threadgroup_memory_length(0, 8192);
                let tg_size = MTLSize::new(32, 4, 1);
                let tg_count = MTLSize::new(
                    ((n + 31) / 32) as u64,
                    ((m + 63) / 64) as u64,
                    1,
                );
                enc.dispatch_thread_groups(tg_count, tg_size);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }

        let mut my_total = 0u64;
        let mut llama_total = 0u64;
        for _ in 0..30 {
            // Self
            let output = device.new_buffer(
                (m * n * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
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
            enc.set_buffer(3, Some(&output), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            let tg_size = MTLSize::new(32, 4, 1);
            let tg_count =
                MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            let start = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            my_total += start.elapsed().as_micros() as u64;
            // llama
            let output = device.new_buffer(
                (m * n * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
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
            enc.set_buffer(3, Some(&output), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            let start = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            llama_total += start.elapsed().as_micros() as u64;
        }
        let my_avg = my_total as f64 / 30.0;
        let llama_avg = llama_total as f64 / 30.0;
        println!(
            "{} | {:.1} | {:.1} | {:.1}%",
            n,
            my_avg,
            llama_avg,
            my_avg / llama_avg * 100.0
        );
    }
}
