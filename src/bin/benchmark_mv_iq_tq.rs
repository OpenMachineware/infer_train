// Benchmark IQ/TQ MV kernels with GPU warmup
// Run: cargo run --release --bin benchmark_mv_iq_tq

use metal::{Device, MTLSize, CompileOptions};

fn main() {
    println!("=== IQ/TQ MV Performance Test ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // ========== 阶段1: GPU预热 ==========
    println!("阶段1: GPU预热 (100次迭代)...");
    
    let warmup_shader = include_str!("../../shaders/mv_iq_tq.metal");
    let iq_grid_tables = include_str!("../../shaders/iq_grid_tables.h");
    let warmup_combined = warmup_shader.replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
    
    let warmup_options = CompileOptions::new();
    warmup_options.set_fast_math_enabled(true);
    let warmup_lib = device.new_library_with_source(&warmup_combined, &warmup_options).unwrap();
    let warmup_fn = warmup_lib.get_function("kernel_mul_mv_iq4_nl_f32", None).unwrap();
    let warmup_pipe = device.new_compute_pipeline_state_with_function(&warmup_fn).unwrap();

    // Warmup kernel params
    let m = 4096usize;
    let k = 4096usize;
    let block_size = 32; // IQ4_NL block size
    let num_blocks = k / block_size;
    let weights_size = m * num_blocks * block_size / 2; // 4-bit
    
    let weights = device.new_buffer(weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer((k * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer((m * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    
    #[repr(C)]
    struct Args { ne00: u32, ne01: u32, nb01: u64 }
    let args = Args { ne00: k as u32, ne01: m as u32, nb01: (num_blocks * block_size / 2) as u64 };
    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new((m / 4) as u64, 1, 1);
    let tg = MTLSize::new(32, 2, 1);

    // 100 warmup iterations
    for i in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&warmup_pipe);
        enc.set_buffer(0, Some(&weights), 0);
        enc.set_buffer(1, Some(&input), 0);
        enc.set_buffer(2, Some(&output), 0);
        enc.set_buffer(3, Some(&args_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        
        if i % 20 == 19 {
            println!("  预热进度: {}/100", i + 1);
        }
    }
    println!("预热完成\n");

    // ========== 阶段2: 测试所有IQ/TQ格式 ==========
    println!("阶段2: 基准测试\n");

    // Format: (name, block_size, kernel_name, weights_per_block)
    let formats = [
        ("IQ4_NL", 32, "kernel_mul_mv_iq4_nl_f32", 16),      // 4-bit, 16 bytes per 32 values
        ("IQ4_XS", 64, "kernel_mul_mv_iq4_xs_f32", 68),      // 4-bit, 68 bytes per 64 values
        ("IQ3_XXS", 32, "kernel_mul_mv_iq3_xxs_f32", 56),    // 3-bit variant
        ("IQ3_S", 32, "kernel_mul_mv_iq3_s_f32", 64),        // 3-bit variant
        ("IQ2_XXS", 32, "kernel_mul_mv_iq2_xxs_f32", 48),    // 2-bit variant
        ("IQ2_XS", 32, "kernel_mul_mv_iq2_xs_f32", 56),      // 2-bit variant
        ("IQ2_S", 32, "kernel_mul_mv_iq2_s_f32_split0", 64),      // 2-bit variant
        ("IQ1_S", 32, "kernel_mul_mv_iq1_s_f32", 40),        // 1-bit variant
        ("IQ1_M", 32, "kernel_mul_mv_iq1_m_f32", 44),        // 1-bit variant
        ("TQ2_0", 32, "kernel_mul_mv_tq2_0_f32", 16),        // 2-bit, 16 bytes per 32 values
        ("TQ1_0", 32, "kernel_mul_mv_tq1_0_f32", 8),         // 1-bit, 8 bytes per 32 values
    ];

    println!("┌─────────┬──────────┬──────────┬──────────┐");
    println!("│ Format  │ GFLOPS   │ vs 87    │ Status   │");
    println!("├─────────┼──────────┼──────────┼──────────┤");

    for (name, block_size, kernel_name, bytes_per_block) in formats.iter() {
        let gflops = match benchmark_format(&device, &queue, m, k, *block_size, *bytes_per_block, kernel_name) {
            Ok(g) => g,
            Err(e) => {
                println!("│ {:7} │ ERROR: {} |", name, e);
                continue;
            }
        };
        
        let ratio = gflops / 87.0 * 100.0;
        let status = if ratio >= 100.0 { "✓ EXCEED" } else if ratio >= 95.0 { "~ MATCH" } else { "✗ BELOW" };

        println!("│ {:7} │ {:8.2} │ {:7.1}% │ {:8} │", name, gflops, ratio, status);
    }

    println!("└─────────┴──────────┴──────────┴──────────┘");
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
    // Compile kernel
    let shader = include_str!("../../shaders/mv_iq_tq.metal");
    let iq_grid_tables = include_str!("../../shaders/iq_grid_tables.h");
    let combined = shader.replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
    
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let lib = device.new_library_with_source(&combined, &opts)
        .map_err(|e| format!("Compile error: {}", e))?;
    let func = lib.get_function(kernel_name, None)
        .map_err(|e| format!("Function error: {}", e))?;
    let pipe = device.new_compute_pipeline_state_with_function(&func)
        .map_err(|e| format!("Pipeline error: {}", e))?;

    // Allocate buffers
    let num_blocks = k / block_size;
    let weights_size = m * num_blocks * bytes_per_block;
    
    let weights = device.new_buffer(weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer((k * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer((m * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct Args { ne00: u32, ne01: u32, nb01: u64 }
    let args = Args { 
        ne00: k as u32, 
        ne01: m as u32, 
        nb01: (num_blocks * bytes_per_block) as u64 
    };
    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new((m / 4) as u64, 1, 1);
    let tg = MTLSize::new(32, 2, 1);

    // Warmup (20 iterations)
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

    // Benchmark (30 iterations)
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