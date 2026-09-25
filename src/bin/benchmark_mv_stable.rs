// GPU warmup + stable benchmark
// Run: cargo run --release --bin benchmark_mv_stable

use metal::{Device, MTLSize, CompileOptions};

fn main() {
    println!("=== GPU预热 + 稳定性能测试 ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // ========== 阶段1: GPU预热 ==========
    println!("阶段1: GPU预热中...");
    let warmup_shader = include_str!("../../shaders/mv_q4_k.metal");
    let warmup_options = CompileOptions::new();
    warmup_options.set_fast_math_enabled(true);
    let warmup_lib = device.new_library_with_source(warmup_shader, &warmup_options).unwrap();
    let warmup_fn = warmup_lib.get_function("kernel_mul_mv_q4_K_f32", None).unwrap();
    let warmup_pipe = device.new_compute_pipeline_state_with_function(&warmup_fn).unwrap();

    // 大量预热迭代
    let m = 4096usize;
    let k = 4096usize;
    let block_size = 144;
    let num_blocks = (k + 255) / 256;
    let weights_size = m * num_blocks * block_size;
    
    let weights = device.new_buffer(weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer((k * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer((m * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    
    #[repr(C)]
    struct Args { ne00: u32, ne01: u32, nb01: u64 }
    let args = Args { ne00: k as u32, ne01: m as u32, nb01: (num_blocks * block_size) as u64 };
    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new((m / 4) as u64, 1, 1);
    let tg = MTLSize::new(32, 2, 1);

    // 100次预热迭代
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

    println!("预热完成，GPU已热\n");
    
    // ========== 阶段2: 稳定基准测试 ==========
    println!("阶段2: 稳定基准测试\n");

    let formats = [
        ("Q2_K", 144, "kernel_mul_mv_q2_K_f32", "shaders/mv_q2_k.metal"),
        ("Q3_K", 128, "kernel_mul_mv_q3_K_f32", "shaders/mv_q3_k.metal"),
        ("Q4_K", 144, "kernel_mul_mv_q4_K_f32", "shaders/mv_q4_k.metal"),
        ("Q5_K", 176, "kernel_mul_mv_q5_K_f32", "shaders/mv_q5_k.metal"),
        ("Q6_K", 210, "kernel_mul_mv_q6_K_f32", "shaders/mv_q6_k.metal"),
    ];

    println!("┌─────────┬──────────┬──────────┬──────────┐");
    println!("│ Format  │ GFLOPS   │ vs 87    │ Status   │");
    println!("├─────────┼──────────┼──────────┼──────────┤");

    for (name, bs, kernel, shader) in formats.iter() {
        let src = std::fs::read_to_string(shader).unwrap();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(true);
        let lib = device.new_library_with_source(&src, &opts).unwrap();
        let func = lib.get_function(kernel, None).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

        let gflops = benchmark(&device, &queue, &pipe, m, k, *bs);
        let ratio = gflops / 87.0 * 100.0;
        let status = if ratio >= 100.0 { "✓ EXCEED" } else if ratio >= 95.0 { "~ MATCH" } else { "✗ BELOW" };

        println!("│ {:7} │ {:8.2} │ {:7.1}% │ {:8} │", name, gflops, ratio, status);
    }

    println!("└─────────┴──────────┴──────────┴──────────┘");
}

fn benchmark(
    device: &Device,
    queue: &metal::CommandQueue,
    pipe: &metal::ComputePipelineState,
    m: usize,
    k: usize,
    bs: usize,
) -> f64 {
    let nb = (k + 255) / 256;
    let ws = m * nb * bs;
    
    let weights = device.new_buffer(ws as u64, metal::MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer((k * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer((m * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct Args { ne00: u32, ne01: u32, nb01: u64 }
    let args = Args { ne00: k as u32, ne01: m as u32, nb01: (nb * bs) as u64 };
    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new((m / 4) as u64, 1, 1);
    let tg = MTLSize::new(32, 2, 1);

    // 20次预热 + 30次测量
    for _ in 0..20 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
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
        enc.set_compute_pipeline_state(pipe);
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
    2.0 * m as f64 * k as f64 * 30.0 / elapsed / 1e9
}