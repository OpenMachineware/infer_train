use metal::*;
use std::ffi::c_void;

const QK_K: usize = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BlockQ2K {
    d: [half::f16; 2], // scales
    qs: [u8; 128],     // quantized values
    scales: [u8; 16],  // scales
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Read our template shader
    let my_shader = include_str!("../../shaders/mul_mv_template.metal");

    // Read llama.cpp shader
    let llama_shader = include_str!("/tmp/q4_k_standalone.metal");

    // Create a simple Q2_K block with known values
    let m = 1; // single row
    let k = 256; // single block
    let nb = k / QK_K;

    let mut weights_data = vec![0u8; std::mem::size_of::<BlockQ2K>()];

    // Fill with simple pattern
    for i in 0..weights_data.len() {
        weights_data[i] = (i % 256) as u8;
    }

    let input: Vec<f32> = (0..k).map(|i| ((i % 10) as f32) * 0.1).collect();

    // Compile both shaders
    let compile_options = CompileOptions::new();
    let my_lib =
        device.new_library_with_source(my_shader, &compile_options).unwrap();

    // Get Q2_K kernel
    let kernel_name = "kernel_mul_mv_q2_K_f32";
    let cv = create_function_constants(2, 1, 1, 1);
    let my_func = my_lib.get_function(kernel_name, Some(cv.clone())).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    let weights_buf = device.new_buffer_with_data(
        weights_data.as_ptr() as *const _,
        weights_data.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct MulMVArgs {
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

    let args = MulMVArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: std::mem::size_of::<BlockQ2K>() as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let grid = MTLSize::new(1, 1, 1);
    let tg = MTLSize::new(64, 1, 1); // NSG=2, 32 threads per SIMD

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&my_pipeline);
    encoder.set_bytes(
        0,
        std::mem::size_of::<MulMVArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    encoder.set_buffer(1, Some(&weights_buf), 0);
    encoder.set_buffer(2, Some(&input_buf), 0);
    encoder.set_buffer(3, Some(&output_buf), 0);
    encoder.dispatch_thread_groups(grid, tg);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let output = unsafe {
        std::slice::from_raw_parts(output_buf.contents() as *const f32, m)
    };
    println!("Q2_K output: {:?}", output);
}

fn create_function_constants(
    nsg: i16,
    ne12: i32,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    let fc_nsg: c_int = 600;
    let fc_ne12: c_int = 602;
    let fc_r2: c_int = 603;
    let fc_r3: c_int = 604;

    unsafe {
        cv.set_constant_value(
            &fc_nsg as *const _ as *const c_void,
            MTLDataType::Short,
            &nsg as *const _ as *const c_void,
        );
        cv.set_constant_value(
            &fc_ne12 as *const _ as *const c_void,
            MTLDataType::Int,
            &ne12 as *const _ as *const c_void,
        );
        cv.set_constant_value(
            &fc_r2 as *const _ as *const c_void,
            MTLDataType::Short,
            &r2 as *const _ as *const c_void,
        );
        cv.set_constant_value(
            &fc_r3 as *const _ as *const c_void,
            MTLDataType::Short,
            &r3 as *const _ as *const c_void,
        );
    }
    cv
}
