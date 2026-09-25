// Benchmark all N ranges to find performance gaps
use metal::{Device, MTLSize, CompileOptions, FunctionDescriptor, FunctionConstantValues, MTLDataType};

fn main() {
    println!("=== All N Range Performance ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // Compile template kernel with function constants
    let template_shader = include_str!("../../shaders/mul_mm_standalone.metal");
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let template_lib = device.new_library_with_source(template_shader, &opts).unwrap();

    // Test different N values
    let n_values = [1, 2, 4, 8, 16, 32, 64, 128];
    let m = 4096usize;
    let k = 4096usize;
    let block_size = 144;

    println!("┌─────┬──────────┬────────────┬──────────┐");
    println!("│  N  │ GFLOPS   │ Kernel     │ Status   │");
    println!("├─────┼──────────┼────────────┼──────────┤");

    // Warmup
    warmup_gpu(&device, &queue);

    for n in n_values {
        let (gflops, kernel_type) = benchmark_n(&device, &queue, &template_lib, m, n, k, block_size);
        let status = if gflops > 1000.0 { "✓ GOOD" } else if gflops > 500.0 { "~ OK" } else { "✗ SLOW" };
        println!("│ {:3} │ {:8.2} │ {:10} │ {:8} │", n, gflops, kernel_type, status);
    }

    println!("└─────┴──────────┴────────────┴──────────┘");
}

fn warmup_gpu(device: &Device, queue: &metal::CommandQueue) {
    // Use template kernel for warmup
    let shader = include_str!("../../shaders/mul_mm_standalone.metal");
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let lib = device.new_library_with_source(shader, &opts).unwrap();

    // Create function constants
    let cv = FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
    unsafe {
        cv.set_constant_value_at_index(&false_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 0);
        cv.set_constant_value_at_index(&false_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 1);
        cv.set_constant_value_at_index(&one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 2);
    }

    let desc = FunctionDescriptor::new();
    desc.set_name("kernel_mul_mm_q4_K_f32");
    desc.set_constant_values(&cv);

    let func = lib.new_function_with_descriptor(&desc).unwrap();
    let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

    // Warmup iterations
    for _ in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        // Minimal dispatch
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 4, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

fn benchmark_n(
    device: &Device,
    queue: &metal::CommandQueue,
    template_lib: &metal::Library,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> (f64, &'static str) {
    // Determine which kernel to use
    if n == 1 {
        // Use MV kernel
        let shader = include_str!("../../shaders/mv_q4_k.metal");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(true);
        let lib = device.new_library_with_source(shader, &opts).unwrap();
        let func = lib.get_function("kernel_mul_mv_q4_K_f32", None).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

        let gflops = benchmark_mv(&device, &queue, &pipe, m, k, block_size);
        (gflops, "MV")
    } else if n >= 32 {
        // Use template GEMM
        let cv = FunctionConstantValues::new();
        let bc_out = m % 64 != 0 || n % 32 != 0;
        let false_val: bool = false;
        let bc_out_val: bool = bc_out;
        let one_val: i16 = 1;
        unsafe {
            cv.set_constant_value_at_index(&false_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 0);
            cv.set_constant_value_at_index(&bc_out_val as *const _ as *const std::ffi::c_void, MTLDataType::Bool, 1);
            cv.set_constant_value_at_index(&one_val as *const _ as *const std::ffi::c_void, MTLDataType::Short, 2);
        }

        let desc = FunctionDescriptor::new();
        desc.set_name("kernel_mul_mm_q4_K_f32");
        desc.set_constant_values(&cv);

        let func = template_lib.new_function_with_descriptor(&desc).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

        let gflops = benchmark_gemm(&device, &queue, &pipe, m, n, k, block_size);
        (gflops, "Template")
    } else {
        // Use old GEMM (N=2-31)
        let shader = include_str!("../../shaders/gemm_q4_k.metal");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(true);
        let lib = device.new_library_with_source(shader, &opts).unwrap();
        let func = lib.get_function("kernel_gemm_q4_k_f32", None).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&func).unwrap();

        let gflops = benchmark_old_gemm(&device, &queue, &pipe, m, n, k, block_size);
        (gflops, "Old GEMM")
    }
}

fn benchmark_mv(
    device: &Device,
    queue: &metal::CommandQueue,
    pipe: &metal::ComputePipelineState,
    m: usize,
    k: usize,
    block_size: usize,
) -> f64 {
    let num_blocks = (k + 255) / 256;
    let ws = m * num_blocks * block_size;

    let weights = device.new_buffer(ws as u64, metal::MTLResourceOptions::StorageModeShared);
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

    // Warmup
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

    // Benchmark
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

fn benchmark_gemm(
    device: &Device,
    queue: &metal::CommandQueue,
    pipe: &metal::ComputePipelineState,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> f64 {
    let num_blocks = (k + 255) / 256;
    let ws = m * num_blocks * block_size;
    let is = n * k * 2; // FP16
    let os = m * n * 4;

    let weights = device.new_buffer(ws as u64, metal::MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer(is as u64, metal::MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer(os as u64, metal::MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct Args {
        ne00: i32, ne02: i32, nb01: u64, nb02: u64, nb03: u64,
        ne12: i32, nb10: u64, nb11: u64, nb12: u64, nb13: u64,
        ne0: i32, ne1: i32, r2: i16, r3: i16,
    }
    let args = Args {
        ne00: k as i32, ne02: 1, nb01: (num_blocks * block_size) as u64, nb02: 0, nb03: 0,
        ne12: 1, nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
        ne0: m as i32, ne1: n as i32, r2: 1, r3: 1,
    };
    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let grid = MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
    let tg = MTLSize::new(32, 4, 1);

    // Warmup + benchmark
    for _ in 0..20 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&args_buf), 0);
        enc.set_buffer(1, Some(&weights), 0);
        enc.set_buffer(2, Some(&input), 0);
        enc.set_buffer(3, Some(&output), 0);
        enc.set_threadgroup_memory_length(4096 + 2048, 0);
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
        enc.set_buffer(0, Some(&args_buf), 0);
        enc.set_buffer(1, Some(&weights), 0);
        enc.set_buffer(2, Some(&input), 0);
        enc.set_buffer(3, Some(&output), 0);
        enc.set_threadgroup_memory_length(4096 + 2048, 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let elapsed = start.elapsed().as_secs_f64();
    2.0 * m as f64 * n as f64 * k as f64 * 30.0 / elapsed / 1e9
}

fn benchmark_old_gemm(
    device: &Device,
    queue: &metal::CommandQueue,
    pipe: &metal::ComputePipelineState,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> f64 {
    // Similar to new GEMM but uses different kernel
    benchmark_gemm(device, queue, pipe, m, n, k, block_size)
}
