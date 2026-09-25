// Verify template kernel correctness
// Run: cargo run --release --bin verify_template_correct

use infer_train::quant::types::BlockQ4K;
use metal::{Device, MTLSize};
use half::f16;

fn main() {
    println!("=== Template Kernel Correctness Verification ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let m = 4096;
    let k = 4096;
    let n = 128;

    println!("M={}, K={}, N={}\n", m, k, n);

    // Create weights with known pattern
    let weights: Vec<BlockQ4K> = (0..m * k / 256).map(|i| {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        // Use simple pattern: all scales = 1.0, all qs = 8
        for j in 0..12 { scales[j] = 64; } // Scale = 1.0 in Q4_K encoding
        for j in 0..128 { qs[j] = (8 + (i % 8)) as u8; }
        BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        }
    }).collect();

    // Create input with simple pattern
    let input_f16: Vec<u16> = (0..k * n)
        .map(|i| half::f16::from_f32(((i % 256) as f32) / 256.0).to_bits())
        .collect();

    // Create buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        (weights.len() * std::mem::size_of::<BlockQ4K>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const std::ffi::c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (m * n * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Compile template kernel
    let shader_source = include_str!("../../shaders/mul_mm_standalone.metal");
    let compile_options = metal::CompileOptions::new();
    let library = device.new_library_with_source(shader_source, &compile_options)
        .expect("Failed to compile shader");
    let kernel = library.get_function("kernel_mul_mm_q4_K_f32", None)
        .expect("Failed to get kernel");
    let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    // Args
    #[repr(C)]
    struct GemmArgs {
        ne00: i32, ne02: i32, nb01: u64, nb02: u64, nb03: u64,
        ne12: i32, nb10: u64, nb11: u64, nb12: u64, nb13: u64,
        ne0: i32, ne1: i32, r2: i16, r3: i16,
    }

    let nb = k / 256;
    let args = GemmArgs {
        ne00: k as i32, ne02: 1,
        nb01: (nb * std::mem::size_of::<BlockQ4K>()) as u64,
        nb02: 0, nb03: 0,
        ne12: 1, nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
        ne0: m as i32, ne1: n as i32, r2: 1, r3: 1,
    };

    let args_buffer = device.new_buffer_with_data(
        &args as *const GemmArgs as *const std::ffi::c_void,
        std::mem::size_of::<GemmArgs>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Run kernel
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&args_buffer), 0);
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_threadgroup_memory_length(0, 8192);

    // Correct grid: x=4 (columns), y=64 (rows)
    let grid = MTLSize::new(4, 64, 1);
    let threadgroup = MTLSize::new(32, 4, 1);
    encoder.dispatch_thread_groups(grid, threadgroup);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Check output
    let output = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, m * n)
    };

    let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
    let zero_count = output.iter().filter(|&&x| x == 0.0).count();
    let min_val = output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = output.iter().map(|&x| x as f64).sum();

    println!("Output statistics:");
    println!("  Non-zero values: {} / {} ({:.1}%)", non_zero_count, m * n, 100.0 * non_zero_count as f64 / (m * n) as f64);
    println!("  Zero values: {} ({:.1}%)", zero_count, 100.0 * zero_count as f64 / (m * n) as f64);
    println!("  Min: {:.6}", min_val);
    println!("  Max: {:.6}", max_val);
    println!("  Sum: {:.2}", sum);
    println!("  Mean: {:.6}", sum / (m * n) as f64);
    println!();

    // Sample some positions
    println!("Sample values (first 10x10 block):");
    for i in 0..10 {
        for j in 0..10 {
            print!("{:8.2} ", output[i + j * m]);
        }
        println!();
    }
    println!();

    // Check diagonal positions
    println!("Diagonal positions (row=col*32 for column-major):");
    for col in 0..4 {
        let row = col * 32;
        let idx = row + col * m;
        println!("  output[{},{}] = output[{}] = {:.6}", row, col, idx, output[idx]);
    }
}
