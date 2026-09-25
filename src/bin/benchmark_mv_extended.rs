// Extended MV kernel benchmark for stable frequency
use metal::{Device, MTLSize, CompileOptions, MTLResourceOptions};
use std::time::{Duration, Instant};

fn main() {
    println!("=== Extended MV Kernel Benchmark ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    let mv_shader = include_str!("../../shaders/mv_q4_k.metal");
    let library = device.new_library_with_source(mv_shader, &CompileOptions::new())
        .expect("Failed to compile MV shader");

    let function = library.get_function("kernel_mul_mv_q4_K_f32", None).expect("Failed to get function");
    let pipeline = device.new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline");

    let m = 4096;
    let k = 4096;
    let block_size = 144;
    let num_blocks_k = (k + 255) / 256;

    let weights_size = m * num_blocks_k * block_size;
    let input_size = k * 4;
    let output_size = m * 4;

    let weights_buffer = device.new_buffer(weights_size as u64, MTLResourceOptions::StorageModeShared);
    let input_buffer = device.new_buffer(input_size as u64, MTLResourceOptions::StorageModeShared);
    let output_buffer = device.new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

    let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01: (num_blocks_k * block_size) as u64 };
    let args_buffer = device.new_buffer_with_data(
        &args as *const MvArgs as *const std::ffi::c_void,
        std::mem::size_of::<MvArgs>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let nr0 = 2;
    let nsg = 2;
    let rows_per_tg = nr0 * nsg;
    let grid_size = MTLSize::new(((m + rows_per_tg - 1) / rows_per_tg) as u64, 1, 1);
    let threadgroup_size = MTLSize::new(32, nsg as u64, 1);

    // Extended warmup - 100 iterations over several seconds
    println!("Warming up GPU...");
    for _ in 0..100 {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&args_buffer), 0);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    // Measure over 10 seconds for stable frequency
    println!("Measuring performance over 10 seconds...\n");
    let duration = Duration::from_secs(10);
    let start = Instant::now();
    let mut iterations = 0;

    while start.elapsed() < duration {
        for _ in 0..100 {
            let cmd_buffer = queue.new_command_buffer();
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&weights_buffer), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&args_buffer), 0);
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            encoder.end_encoding();
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();
            iterations += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * k as f64 * iterations as f64;
    let gflops = flops / elapsed / 1e9;

    println!("Iterations: {}", iterations);
    println!("Time: {:.2}s", elapsed);
    println!("GFLOPS: {:.2}", gflops);
    println!("vs llama.cpp (87 GFLOPS): {:.1}%", gflops / 87.0 * 100.0);
}