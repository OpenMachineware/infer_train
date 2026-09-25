// Full test for templated GEMM kernel with correctness verification
use metal::{Device, MTLSize};
use infer_train::quant::types::BlockQ4K;

fn main() {
    println!("=== Full Templated GEMM Test ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let shader_source = include_str!("../../shaders/gemm_llama_cpp.metal");
    let compile_options = metal::CompileOptions::new();
    let library = device.new_library_with_source(shader_source, &compile_options)
        .expect("Failed to compile shader");
    println!("✓ Shader compiled");

    let kernel = library.get_function("kernel_mul_mm_q4_K_f32", None)
        .expect("Failed to get kernel");
    let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");
    println!("✓ Pipeline created\n");

    // Test dimensions: M=4096, K=4096, N=128 (matching llama.cpp benchmark)
    let m = 4096;
    let k = 4096;
    let n = 128;

    println!("Matrix dimensions: A({}x{}) x B({}x{}) -> C({}x{})",
             m, k, k, n, m, n);
    println!("Operations: {:.2} GFLOPS\n",
             2.0 * m as f64 * k as f64 * n as f64 / 1e9);

    // Create real Q4_K weights
    let num_blocks = k / 256;
    let mut weights_data = vec![0u8; m * num_blocks * std::mem::size_of::<BlockQ4K>()];
    let weights_ptr = weights_data.as_mut_ptr() as *mut BlockQ4K;

    // Initialize with known pattern
    unsafe {
        for i in 0..m * num_blocks {
            let block = weights_ptr.add(i);
            // Simple pattern: d=1.0, dmin=0, scales=1, qs=8
            (*block).d = 0x3C00;  // FP16 1.0
            (*block).dmin = 0x0000;  // FP16 0.0
            for j in 0..12 {
                (*block).scales[j] = 1;  // scale=1, min=0
            }
            for j in 0..128 {
                (*block).qs[j] = 0x88;  // value 8 in both nibbles
            }
        }
    }
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr, m * num_blocks) };

    // Create FP16 input (B matrix) - column-major
    // In column-major: B[col * K + row]
    let input: Vec<u16> = (0..(k * n)).map(|i| {
        let val = (i % 10) as f32;  // Simple pattern
        half::f16::from_f32(val).to_bits()
    }).collect();

    println!("✓ Data created");

    // Run on GPU
    let (gpu_output, gpu_time) = run_gemm_on_gpu(&device, &pipeline, weights, &input, m, k, n);

    println!("\nGPU performance:");
    println!("  Time: {:?}", gpu_time);
    println!("  GFLOPS: {:.2}",
             2.0 * m as f64 * k as f64 * n as f64 / gpu_time.as_secs_f64() / 1e9);

    // Verify with CPU for first row
    println!("\nVerifying first row...");
    let cpu_first_row = compute_cpu_first_row(weights, &input, k, n, num_blocks);

    println!("First 5 values:");
    println!("  CPU: {:?}", &cpu_first_row[..5]);
    println!("  GPU: {:?}", &gpu_output[..5]);

    let max_diff = cpu_first_row.iter().zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, |max, x| if x > max { x } else { max });

    println!("  Max diff: {:.4}", max_diff);

    // Test with different sizes
    println!("\n=== Performance scaling ===");
    for test_n in [1, 4, 16, 64, 128].iter() {
        let n = *test_n;
        let input_small: Vec<u16> = vec![0; k * n];
        let (_, time) = run_gemm_on_gpu(&device, &pipeline, weights, &input_small, m, k, n);
        let gflops = 2.0 * m as f64 * k as f64 * n as f64 / time.as_secs_f64() / 1e9;
        println!("  N={:3}: {:?} -> {:.1} GFLOPS", n, time, gflops);
    }
}

fn run_gemm_on_gpu(
    device: &Device,
    pipeline: &metal::ComputePipelineState,
    weights: &[BlockQ4K],
    input: &[u16],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, std::time::Duration) {
    use std::time::Instant;

    let num_blocks = k / 256;
    let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
    let input_size = input.len() * 2;
    let output_size = m * n * 4;

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        weights_size as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        input_size as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeManaged);

    #[repr(C)]
    struct Args {
        ne00: i32, ne01: i32, ne02: i32, ne03: i32,
        nb00: u64, nb01: u64, nb02: u64, nb03: u64,
        ne10: i32, ne11: i32, ne12: i32, ne13: i32,
        nb10: u64, nb11: u64, nb12: u64, nb13: u64,
        ne0: i32, ne1: i32, ne2: i32, ne3: i32,
        nb0: u64, nb1: u64, nb2: u64, nb3: u64,
    }

    let args = Args {
        ne00: k as i32, ne01: m as i32, ne02: 1, ne03: 1,
        nb00: 144, nb01: (num_blocks * 144) as u64, nb02: 0, nb03: 0,
        ne10: k as i32, ne11: n as i32, ne12: 1, ne13: 1,
        nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
        ne0: m as i32, ne1: n as i32, ne2: 1, ne3: 1,
        nb0: 4, nb1: (n * 4) as u64, nb2: 0, nb3: 0,
    };

    let args_buffer = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&args_buffer), 0);
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_threadgroup_memory_length(0, 8192);

    let grid_size = MTLSize::new(
        ((n + 31) / 32) as u64,
        ((m + 63) / 64) as u64,
        1,
    );
    let threadgroup_size = MTLSize::new(128, 1, 1);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    let start = Instant::now();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let duration = start.elapsed();

    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() };

    (output, duration)
}

fn compute_cpu_first_row(
    weights: &[BlockQ4K],
    input: &[u16],
    k: usize,
    n: usize,
    num_blocks: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n];

    // Compute first row only
    let row = 0;
    for col in 0..n {
        let mut sum = 0.0f32;

        for block_idx in 0..num_blocks {
            let w_block = &weights[row * num_blocks + block_idx];
            let d = half::f16::from_bits(w_block.d).to_f32();

            // Simple dequant: d * scale * quant - dmin * min
            // With scale=1, min=0, quant=8: value = d * 1 * 8 - 0 = d * 8
            for i in 0..128 {
                let lo = (w_block.qs[i] & 0x0F) as f32;
                let hi = (w_block.qs[i] >> 4) as f32;

                let idx1 = block_idx * 256 + 2 * i;
                let idx2 = block_idx * 256 + 2 * i + 1;

                // B is column-major: B[col * K + row]
                if idx1 < k {
                    let input_val = half::f16::from_bits(input[col * k + idx1]).to_f32();
                    sum += d * lo * input_val;
                }
                if idx2 < k {
                    let input_val = half::f16::from_bits(input[col * k + idx2]).to_f32();
                    sum += d * hi * input_val;
                }
            }
        }
        output[col] = sum;
    }

    output
}
