// Debug: check data and intermediate results
use half::f16;
use infer_train::quant::types::BlockIQ4NL;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let m = 4usize;
    let k = 32usize;
    let n = 4usize;
    let seed = 42u64; // Minimal size

    // Generate IQ4_NL with non-zero data
    let nb = k / 32;
    let mut weights: Vec<BlockIQ4NL> = Vec::new();
    for i in 0..m * nb {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            // Fill with non-zero values (each byte packs two 4-bit values)
            qs[j] = ((i * 16 + j) % 256) as u8;
        }
        weights.push(BlockIQ4NL {
            d: f16::from_f32(0.5).to_bits(), // Non-zero scale
            qs,
        });
    }

    // Generate non-zero input
    let mut input_f16: Vec<u16> = Vec::new();
    for i in 0..k * n {
        input_f16.push(f16::from_f32(0.1 * (i as f32 + 1.0)).to_bits());
    }

    println!(
        "Weights block[0]: d={:.3}, qs[0..4]={:?}",
        f16::from_bits(weights[0].d).to_f32(),
        &weights[0].qs[0..4]
    );
    println!(
        "Input[0..4]: {:?}",
        input_f16[0..4]
            .iter()
            .map(|&x| f16::from_bits(x).to_f32())
            .collect::<Vec<_>>()
    );

    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/blocks.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../src/ops/metal/iq_grid_tables.h");
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
            "src/ops/metal/mul_mm_template.metallib",
        ))
        .unwrap();

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
        nb01: 18,
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

    println!(
        "\nArgs: ne00={}, ne01={}, ne10={}, ne11={}, ne0={}, ne1={}",
        args.ne00, args.ne01, args.ne10, args.ne11, args.ne0, args.ne1
    );

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
                .get_function("kernel_mul_mm_iq4_nl_f32", Some(cv.clone()))
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

    // Run both
    for (pipeline, output) in
        [(&my_pipeline, &output_my), (&llama_pipeline, &output_llama)]
    {
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
        let tg_count =
            MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let my_data = unsafe {
        std::slice::from_raw_parts(output_my.contents() as *const f32, m * n)
    };
    let llama_data = unsafe {
        std::slice::from_raw_parts(output_llama.contents() as *const f32, m * n)
    };

    println!("\nSelf output: {:?}", &my_data[0..4]);
    println!("Llama output: {:?}", &llama_data[0..4]);

    let non_zero_self = my_data.iter().filter(|&&x| x.abs() > 0.0001).count();
    let non_zero_llama =
        llama_data.iter().filter(|&&x| x.abs() > 0.0001).count();
    println!(
        "\nNon-zero elements: self={}, llama={}",
        non_zero_self, non_zero_llama
    );
}
