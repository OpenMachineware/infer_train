// Benchmark Q2_K/Q3_K with optimized dequantization
// Run: cargo run --release --bin benchmark_q2_q3_opt

use metal::{Device, MTLSize, CompileOptions, FunctionDescriptor, FunctionConstantValues, MTLDataType};

fn main() {
    println!("=== Q2_K/Q3_K Optimization Comparison ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let m = 4096;
    let k = 4096;
    let n = 128;

    println!("M={}, K={}, N={}\n", m, k, n);

    // Compile original shader
    let orig_shader = include_str!("../../shaders/mul_mm_standalone.metal");
    let compile_options = CompileOptions::new();
    let orig_library = device.new_library_with_source(orig_shader, &compile_options)
        .expect("Failed to compile original shader");

    // Create function constants
    let constant_values = FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
    
    unsafe {
        constant_values.set_constant_value_at_index(
            &false_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 0);
        constant_values.set_constant_value_at_index(
            &false_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 1);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 2);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 3);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 4);
        constant_values.set_constant_value_at_index(
            &one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 5);
    }

    println!("┌────────┬──────────┬───────────┬──────────┐");
    println!("│ Format │ GFLOPS   │ vs llama  │ Coverage │");
    println!("├────────┼──────────┼───────────┼──────────┤");

    for (name, kernel_name, block_size, llama_gflops) in [
        ("Q2_K", "kernel_mul_mm_q2_K_f32", 144, 2738.79),
        ("Q3_K", "kernel_mul_mm_q3_K_f32", 128, 3552.50),
    ] {
        let descriptor = FunctionDescriptor::new();
        descriptor.set_name(kernel_name);
        descriptor.set_constant_values(&constant_values);
        
        let function = orig_library.new_function_with_descriptor(&descriptor)
            .expect("Failed to create function");
        let pipeline = device.new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create pipeline");

        // Create buffers
        let weights_buffer = create_weights_buffer(&device, m, k, block_size);
        let input_buffer = create_input_buffer(&device, k, n);
        let output_buffer = device.new_buffer(
            (m * n * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

        // Args
        #[repr(C)]
        struct GemmArgs {
            ne00: i32, ne02: i32, nb01: u64, nb02: u64, nb03: u64,
            ne12: i32, nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, r2: i16, r3: i16,
        }

        let args = GemmArgs {
            ne00: k as i32, ne02: 1, nb01: (k/256 * block_size) as u64, nb02: 0, nb03: 0,
            ne12: 1, nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
            ne0: m as i32, ne1: n as i32, r2: 1, r3: 1,
        };
        let args_buffer = device.new_buffer_with_data(
            &args as *const GemmArgs as *const std::ffi::c_void,
            std::mem::size_of::<GemmArgs>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Extended warmup (10 iterations to stabilize GPU)
        for _ in 0..10 {
            run_gemm(&queue, &pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer);
        }

        // Benchmark (15 iterations)
        let n_iter = 15;
        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            run_gemm(&queue, &pipeline, &args_buffer, &weights_buffer, &input_buffer, &output_buffer);
        }
        let time = start.elapsed().as_micros() as f64 / n_iter as f64;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / (time * 1e-6) / 1e9;

        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, m * n)
        };
        let coverage = output.iter().filter(|&&x| x != 0.0).count() as f64 / (m * n) as f64 * 100.0;

        println!("│ {:6} │ {:8.2} │ {:7.1}%  │ {:7.1}% │", 
            name, gflops, gflops / llama_gflops * 100.0, coverage);
    }

    println!("└────────┴──────────┴───────────┴──────────┘");
    println!();
    println!("llama.cpp reference: Q2_K=2739, Q3_K=3552 GFLOPS");
}

fn create_weights_buffer(device: &Device, m: usize, k: usize, block_size: usize) -> metal::Buffer {
    let n_blocks = m * k / 256;
    let data: Vec<u8> = (0..n_blocks * block_size).map(|i| ((i * 17 + 13) % 256) as u8).collect();
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        (data.len()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn create_input_buffer(device: &Device, k: usize, n: usize) -> metal::Buffer {
    use half::f16;
    let data: Vec<u16> = (0..k * n)
        .map(|i| f16::from_f32((i % 10) as f32 * 0.1).to_bits())
        .collect();
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        (data.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn run_gemm(
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    args: &metal::Buffer,
    weights: &metal::Buffer,
    input: &metal::Buffer,
    output: &metal::Buffer,
) {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(args), 0);
    enc.set_buffer(1, Some(weights), 0);
    enc.set_buffer(2, Some(input), 0);
    enc.set_buffer(3, Some(output), 0);
    enc.set_threadgroup_memory_length(0, 8192);
    enc.dispatch_thread_groups(MTLSize::new(4, 64, 1), MTLSize::new(32, 4, 1));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}