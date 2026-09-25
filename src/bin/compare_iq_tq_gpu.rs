// Fair comparison: Our IQ/TQ GPU kernels vs llama.cpp
// Run: cargo run --release --bin compare_iq_tq_gpu

use metal::{Device, MTLSize, CompileOptions};
use std::process::Command;

fn main() {
    println!("=== IQ/TQ GPU Fair Comparison ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // ========== 阶段1: GPU预热 ==========
    println!("阶段1: GPU预热...");
    warmup_gpu(&device, &queue);
    println!("预热完成\n");

    // ========== 阶段2: 我们的实现 ==========
    println!("阶段2: 测试我们的Rust实现...\n");

    let our_results = benchmark_our_implementation(&device, &queue);

    // ========== 阶段3: llama.cpp基线 ==========
    println!("\n阶段3: 测试llama.cpp实现...");
    println!("(llama.cpp没有单独的IQ/TQ GPU基准测试)");
    println!("使用CPU NEON基准作为参考:\n");

    // 从记忆中获取llama.cpp CPU基准
    let llama_cpp_baselines = [
        ("IQ4_NL", 210.52),
        ("IQ4_XS", 55.89),
        ("IQ1_S", 23.48),
        ("IQ1_M", 21.33),
        ("IQ2_XXS", 26.85),
        ("IQ2_XS", 30.14),
        ("IQ2_S", 20.32),
        ("IQ3_XXS", 22.03),
        ("IQ3_S", 19.64),
        ("TQ1_0", 68.05),
        ("TQ2_0", 100.44),
    ];

    println!("┌─────────┬──────────┬──────────┬─────────┐");
    println!("│ Format  │ GPU(Mine)│ CPU(llama│ Note    │");
    println!("├─────────┼──────────┼──────────┼─────────┤");

    for (name, our_gflops) in &our_results {
        let llama_gflops = llama_cpp_baselines.iter()
            .find(|(n, _)| n == name)
            .map(|(_, g)| *g)
            .unwrap_or(0.0);

        println!("│ {:7} │ {:8.2} │ {:8.2} │ GPU vs CPU│",
            name, our_gflops, llama_gflops);
    }

    println!("└─────────┴──────────┴──────────┴─────────┘");
    println!("\n注意: GPU和CPU架构不同，不能直接对比性能");
}

fn warmup_gpu(device: &Device, queue: &metal::CommandQueue) {
    let shader = include_str!("../../shaders/mv_q4_k.metal");
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let lib = device.new_library_with_source(shader, &opts).unwrap();
    let func = lib.get_function("kernel_mul_mv_q4_K_f32", None).unwrap();
    let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

    let m = 4096usize;
    let k = 4096usize;
    let nb = (k + 255) / 256;
    let ws = m * nb * 144;

    let weights = device.new_buffer(ws as u64, metal::MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer((k * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer((m * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct Args { ne00: u32, ne01: u32, nb01: u64 }
    let args = Args { ne00: k as u32, ne01: m as u32, nb01: (nb * 144) as u64 };
    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new((m / 4) as u64, 1, 1);
    let tg = MTLSize::new(32, 2, 1);

    for _ in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&weights), 0);
        enc.set_buffer(1, Some(&input), 0);
        enc.set_buffer(2, Some(&output), 0);
        enc.set_buffer(3, Some(&args_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

fn benchmark_our_implementation(device: &Device, queue: &metal::CommandQueue) -> Vec<(&'static str, f64)> {
    let formats = [
        ("IQ4_NL", 32, "kernel_mul_mv_iq4_nl_f32", 16),
        ("IQ4_XS", 64, "kernel_mul_mv_iq4_xs_f32", 68),
        ("IQ3_XXS", 32, "kernel_mul_mv_iq3_xxs_f32", 56),
        ("IQ3_S", 32, "kernel_mul_mv_iq3_s_f32", 64),
        ("IQ2_XXS", 32, "kernel_mul_mv_iq2_xxs_f32", 48),
        ("IQ2_XS", 32, "kernel_mul_mv_iq2_xs_f32", 56),
        ("IQ2_S", 32, "kernel_mul_mv_iq2_s_f32_split0", 64),
        ("IQ1_S", 32, "kernel_mul_mv_iq1_s_f32", 40),
        ("IQ1_M", 32, "kernel_mul_mv_iq1_m_f32", 44),
        ("TQ2_0", 32, "kernel_mul_mv_tq2_0_f32", 16),
        ("TQ1_0", 32, "kernel_mul_mv_tq1_0_f32", 8),
    ];

    let mut results = Vec::new();

    for (name, block_size, kernel_name, bytes_per_block) in formats.iter() {
        let gflops = match benchmark_format(device, queue, 4096, 4096, *block_size, *bytes_per_block, kernel_name) {
            Ok(g) => g,
            Err(_) => continue,
        };
        results.push((*name, gflops));
        println!("{}: {:.2} GFLOPS", name, gflops);
    }

    results
}

fn benchmark_format(
    device: &Device,
    queue: &metal::CommandQueue,
    m: usize,
    k: usize,
    block_size: usize,
    bytes_per_block: usize,
    kernel_name: &str,
) -> Result<f64, String> {
    let shader = include_str!("../../shaders/mv_iq_tq.metal");
    let iq_grid_tables = include_str!("../../shaders/iq_grid_tables.h");
    let combined = shader.replace("#include \"iq_grid_tables.h\"", iq_grid_tables);

    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let lib = device.new_library_with_source(&combined, &opts)?;
    let func = lib.get_function(kernel_name, None)?;
    let pipe = device.new_compute_pipeline_state_with_function(&func)?;

    let num_blocks = k / block_size;
    let weights_size = m * num_blocks * bytes_per_block;

    let weights = device.new_buffer(weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer((k * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer((m * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct Args { ne00: u32, ne01: u32, nb01: u64 }
    let args = Args { ne00: k as u32, ne01: m as u32, nb01: (num_blocks * bytes_per_block) as u64 };
    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new((m / 4) as u64, 1, 1);
    let tg = MTLSize::new(32, 2, 1);

    for _ in 0..20 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&weights), 0);
        enc.set_buffer(1, Some(&input), 0);
        enc.set_buffer(2, Some(&output), 0);
        enc.set_buffer(3, Some(&args_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let start = std::time::Instant::now();
    for _ in 0..30 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&weights), 0);
        enc.set_buffer(1, Some(&input), 0);
        enc.set_buffer(2, Some(&output), 0);
        enc.set_buffer(3, Some(&args_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let elapsed = start.elapsed().as_secs_f64();
    Ok(2.0 * m as f64 * k as f64 * 30.0 / elapsed / 1e9)
}
