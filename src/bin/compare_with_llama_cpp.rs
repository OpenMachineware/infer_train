// Compare template kernel with llama.cpp using identical test data
// Run: cargo run --release --bin compare_with_llama_cpp

use infer_train::quant::types::BlockQ4K;
use metal::{Device, MTLSize};
use half::f16;

fn main() {
    println!("=== Template Kernel vs llama.cpp Comparison ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let m = 4096;
    let k = 4096;
    let n = 128;

    println!("M={}, K={}, N={}\n", m, k, n);

    // Create weights - match llama.cpp's test pattern
    let weights: Vec<BlockQ4K> = (0..m * k / 256).map(|block_idx| {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        
        // Initialize with llama.cpp pattern: (i * 17 + 13) % 256
        for j in 0..12 { 
            let val = ((block_idx * 17 + j + 13) % 256) as u8;
            scales[j] = val; 
        }
        for j in 0..128 { 
            let val = ((block_idx * 17 + j + 13) % 256) as u8;
            qs[j] = val; 
        }
        
        // Set d and dmin
        BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        }
    }).collect();

    // Create input - match llama.cpp's test pattern
    let input_f32: Vec<f32> = (0..k * n)
        .map(|i| (i % 10) as f32 * 0.1)
        .collect();
    let input_f16: Vec<u16> = input_f32.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();

    println!("Weights: {} blocks, {} bytes", weights.len(), weights.len() * 144);
    println!("Input: {} floats (converted to FP16)\n", input_f32.len());

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

    // Warmup
    for _ in 0..2 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&args_buffer), 0);
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_buffer), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        let grid = MTLSize::new(4, 64, 1);  // x=4 columns, y=64 rows
        let tg = MTLSize::new(32, 4, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Benchmark
    let n_iter = 10;
    let mut times = Vec::new();
    
    for _ in 0..n_iter {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&args_buffer), 0);
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_buffer), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        let grid = MTLSize::new(4, 64, 1);  // x=4 columns, y=64 rows
        let tg = MTLSize::new(32, 4, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        
        let start = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed().as_micros() as f64);
    }

    let avg_time = times.iter().sum::<f64>() / n_iter as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / (avg_time * 1e-6) / 1e9;

    println!("=== Performance Results ===");
    println!("Grid: 4 x 64 threadgroups (x=columns, y=rows)");
    println!("Threadgroup: 32 x 4 = 128 threads (32 width, 4 simdgroups)");
    println!();
    println!("Average time: {:.2} µs", avg_time);
    println!("GFLOPS: {:.2}", gflops);
    println!("llama.cpp Q4_K: ~3552 GFLOPS");
    println!("Ratio: {:.1}%", gflops / 3552.0 * 100.0);
    println!();

    // Check output coverage
    let output = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, m * n)
    };

    let non_zero = output.iter().filter(|&&x| x != 0.0).count();
    println!("=== Output Coverage ===");
    println!("Non-zero: {} / {} ({:.1}%)", non_zero, m * n, 100.0 * non_zero as f64 / (m * n) as f64);

    // Sample values
    println!("\n=== Sample Output Values ===");
    println!("First 5 values: {:.2} {:.2} {:.2} {:.2} {:.2}", 
        output[0], output[1], output[2], output[3], output[4]);
    println!("Last 5 values: {:.2} {:.2} {:.2} {:.2} {:.2}",
        output[m*n-5], output[m*n-4], output[m*n-3], output[m*n-2], output[m*n-1]);
}