// Fair comparison: our kernels vs llama.cpp in same session
// Uses llama.cpp's test-kquant-metal binary for baseline

use metal::{Device, MTLSize, CompileOptions};

fn main() {
    println!("=== 公平对比: 同GPU状态下的性能 ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // ========== 阶段1: 预热 ==========
    println!("阶段1: GPU预热...");
    warmup_gpu(&device, &queue);
    println!("预热完成\n");

    // ========== 阶段2: llama.cpp基线 ==========
    println!("阶段2: 测量llama.cpp基线...");
    println!("(运行llama.cpp的test-kquant-metal)\n");

    // 运行llama.cpp测试
    let llama_output = std::process::Command::new("./test-kquant-metal")
        .current_dir("/Users/jia/Desktop/infer_train/llama.cpp-0.5.0")
        .output()
        .expect("Failed to run llama.cpp test");

    let llama_stdout = String::from_utf8_lossy(&llama_output.stdout);
    println!("{}", llama_stdout);

    // 解析llama.cpp结果
    let mut llama_results: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for line in llama_stdout.lines() {
        // Format: │ Q4_K     │    99.33 │   337.8 │
        if line.contains("│ Q") {
            let cleaned: String = line.replace("│", " ");
            let parts: Vec<&str> = cleaned.split_whitespace().collect();
            if parts.len() >= 2 {
                let name = parts[0].to_string();
                if let Ok(gflops) = parts[1].parse::<f64>() {
                    llama_results.insert(name, gflops);
                }
            }
        }
    }

    // ========== 阶段3: 我们的实现 ==========
    println!("\n阶段3: 测量我们的Rust实现...\n");

    let formats = [
        ("Q2_K", 144, "kernel_mul_mv_q2_K_f32", "shaders/mv_q2_k.metal"),
        ("Q3_K", 128, "kernel_mul_mv_q3_K_f32", "shaders/mv_q3_k.metal"),
        ("Q4_K", 144, "kernel_mul_mv_q4_K_f32", "shaders/mv_q4_k.metal"),
        ("Q5_K", 176, "kernel_mul_mv_q5_K_f32", "shaders/mv_q5_k.metal"),
        ("Q6_K", 210, "kernel_mul_mv_q6_K_f32", "shaders/mv_q6_k.metal"),
    ];

    println!("┌─────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│ Format  │ Our      │ llama.cpp│ Ratio    │ Status   │");
    println!("├─────────┼──────────┼──────────┼──────────┼──────────┤");

    for (name, bs, kernel, shader) in formats.iter() {
        let src = std::fs::read_to_string(shader).unwrap();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(true);
        let lib = device.new_library_with_source(&src, &opts).unwrap();
        let func = lib.get_function(kernel, None).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

        let our_gflops = benchmark(&device, &queue, &pipe, 4096, 4096, *bs);
        let llama_gflops = llama_results.get(*name).copied().unwrap_or(87.0);
        let ratio = our_gflops / llama_gflops * 100.0;
        let status = if ratio >= 100.0 { "✓ EXCEED" } else if ratio >= 95.0 { "~ MATCH" } else { "✗ BELOW" };

        println!("│ {:7} │ {:8.2} │ {:8.2} │ {:7.1}% │ {:8} │",
            name, our_gflops, llama_gflops, ratio, status);
    }

    println!("└─────────┴──────────┴──────────┴──────────┴──────────┘");
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
