// Test with larger size to avoid boundary issues
use half::f16;
use infer_train::quant::types::BlockIQ4NL;
use metal::{Device, MTLSize};
use std::ffi::c_void;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let m = 128usize;
    let k = 512usize;
    let n = 128usize; // Large enough for tiles

    // Generate IQ4_NL with simple pattern
    let nb = k / 32;
    let mut weights: Vec<BlockIQ4NL> = Vec::new();
    for i in 0..m * nb {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i + j) % 256) as u8;
        }
        weights.push(BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs });
    }

    // Generate input with simple pattern
    let mut input_f16: Vec<u16> = Vec::new();
    for i in 0..k * n {
        input_f16.push(f16::from_f32(0.01 * (i as f32 % 100.0)).to_bits());
    }

    let gemm_source = include_str!("../src/ops/metal/gemm.metal");
    let ggml_common = include_str!("../src/ops/metal/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../src/ops/metal/iq_grid_tables.h");
    let iq_extra_tables = r#"
constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
constexpr constant static int8_t ksigns_iq2xs[128] = {1,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,1,1,1,1,-1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,1,1,1,1,-1,1,-1,1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,};
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

    println!("Self[0..4]: {:?}", &my_data[0..4]);
    println!("Llama[0..4]: {:?}", &llama_data[0..4]);

    let non_zero_self = my_data.iter().filter(|&&x| x.abs() > 0.001).count();
    let non_zero_llama = llama_data
        .iter()
        .filter(|&&x| x.is_finite() && x.abs() > 0.001)
        .count();
    let nan_llama = llama_data.iter().filter(|&&x| x.is_nan()).count();

    println!("\nStats:");
    println!("  Self non-zero: {}/{}", non_zero_self, m * n);
    println!("  Llama non-zero: {}/{}", non_zero_llama, m * n);
    println!("  Llama NaN: {}/{}", nan_llama, m * n);
}
