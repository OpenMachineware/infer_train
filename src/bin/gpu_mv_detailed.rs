// Detailed instrumented GPU MV - measure each step
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

// Function constant indices for MV kernels (from llama.cpp)
const FC_MUL_MV_NSG: NSUInteger = 600;
const FC_MUL_MV_NE12: NSUInteger = 602;
const FC_MUL_MV_R2: NSUInteger = 603;
const FC_MUL_MV_R3: NSUInteger = 604;

/// Create function constants for MV kernel specialization
fn create_mv_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let nsg_ptr: *const c_void = &nsg as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            nsg_ptr,
            MTLDataType::Short,
            FC_MUL_MV_NSG,
        );
        let ne12_ptr: *const c_void = &ne12 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            ne12_ptr,
            MTLDataType::Short,
            FC_MUL_MV_NE12,
        );
        let r2_ptr: *const c_void = &r2 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            r2_ptr,
            MTLDataType::Short,
            FC_MUL_MV_R2,
        );
        let r3_ptr: *const c_void = &r3 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            r3_ptr,
            MTLDataType::Short,
            FC_MUL_MV_R3,
        );
    }
    cv
}

fn main() {
    println!("=== GPU MV Detailed Instrumentation ===\n");

    let m = 4096usize;
    let k = 4096usize;

    // Generate data
    let weights = generate_q4_k_weights(m, k);
    let input: Vec<f32> = (0..k).map(|i| (i as f32).sin()).collect();

    // 1. Device creation
    let t0 = Instant::now();
    let device = Device::system_default().expect("No Metal device");
    println!(
        "1. Device creation: {:.3} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // 2. Library compilation
    let t1 = Instant::now();
    let source = include_str!("../../shaders/mul_mv_template.metal");
    let compile_options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &compile_options)
        .expect("Failed to compile library");
    println!(
        "2. Library compilation: {:.3} ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );

    // 3. Pipeline creation with function constants
    let t2 = Instant::now();
    let cv = create_mv_function_constants(2, 1, 1, 1);
    let kernel = library
        .get_function("kernel_mul_mv_q4_K_f32", Some(cv))
        .expect("Failed to get kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");
    println!(
        "3. Pipeline creation: {:.3} ms",
        t2.elapsed().as_secs_f64() * 1000.0
    );

    // 4. Queue creation
    let t3 = Instant::now();
    let queue = device.new_command_queue();
    println!(
        "4. Queue creation: {:.3} ms",
        t3.elapsed().as_secs_f64() * 1000.0
    );

    // 5. Buffer creation
    let t4 = Instant::now();
    let weights_size = weights.len() * std::mem::size_of_val(&weights[0]);
    let input_size = input.len() * std::mem::size_of_val(&input[0]);
    let output_size = m * std::mem::size_of::<f32>();

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        input_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    println!(
        "5. Buffer creation: {:.3} ms",
        t4.elapsed().as_secs_f64() * 1000.0
    );

    // 6. Args setup
    let t5 = Instant::now();
    let nb = k / 256;
    let nb01 = (nb * std::mem::size_of::<infer_train::quant::types::BlockQ4K>())
        as u64;

    #[repr(C)]
    struct MvArgs {
        ne00: i32,
        ne01: i32,
        ne02: i32,
        nb00: u64,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        nr0: i32,
        r2: i16,
        r3: i16,
    }

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: k as u64 * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };
    println!("6. Args setup: {:.3} us", t5.elapsed().as_secs_f64() * 1e6);

    // Warmup iteration
    let _ = run_mv(
        &queue,
        &pipeline,
        &weights_buffer,
        &input_buffer,
        &output_buffer,
        unsafe {
            std::slice::from_raw_parts(
                &args as *const MvArgs as *const u8,
                std::mem::size_of::<MvArgs>(),
            )
        },
        m,
    );

    // 7. Detailed single iteration timing
    println!("\n=== Single Iteration Breakdown ===");
    let iterations = 10;
    let mut times = [
        ("Command buffer", 0.0f64),
        ("Encoder creation", 0.0f64),
        ("Pipeline set", 0.0f64),
        ("Args set (idx 0)", 0.0f64),
        ("Weights set (idx 1)", 0.0f64),
        ("Input set (idx 2)", 0.0f64),
        ("Output set (idx 3)", 0.0f64),
        ("Dispatch", 0.0f64),
        ("End encoding", 0.0f64),
        ("Commit", 0.0f64),
        ("Wait", 0.0f64),
        ("Read output", 0.0f64),
    ];

    for _ in 0..iterations {
        let t_cb = Instant::now();
        let command_buffer = queue.new_command_buffer();
        times[0].1 += t_cb.elapsed().as_secs_f64() * 1e6;

        let t_enc = Instant::now();
        let encoder = command_buffer.new_compute_command_encoder();
        times[1].1 += t_enc.elapsed().as_secs_f64() * 1e6;

        let t_pipe = Instant::now();
        encoder.set_compute_pipeline_state(&pipeline);
        times[2].1 += t_pipe.elapsed().as_secs_f64() * 1e6;

        // Template kernel signature: (args, src0, src1, dst)
        // Index 0: args struct
        // Index 1: src0 (weights)
        // Index 2: src1 (input)
        // Index 3: dst (output)
        let t_bytes = Instant::now();
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        times[3].1 += t_bytes.elapsed().as_secs_f64() * 1e6;

        let t_b0 = Instant::now();
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        times[4].1 += t_b0.elapsed().as_secs_f64() * 1e6;

        let t_b1 = Instant::now();
        encoder.set_buffer(2, Some(&input_buffer), 0);
        times[5].1 += t_b1.elapsed().as_secs_f64() * 1e6;

        let t_b2 = Instant::now();
        encoder.set_buffer(3, Some(&output_buffer), 0);
        times[6].1 += t_b2.elapsed().as_secs_f64() * 1e6;

        let t_disp = Instant::now();
        let nsg = 2u64;
        let nr0 = 2u64;
        let rows_per_tg = nsg * nr0;
        let tg_size = MTLSize { width: 32, height: nsg, depth: 1 };
        let tg_count = MTLSize {
            width: (m as u64 + rows_per_tg - 1) / rows_per_tg,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(tg_count, tg_size);
        times[7].1 += t_disp.elapsed().as_secs_f64() * 1e6;

        let t_end = Instant::now();
        encoder.end_encoding();
        times[8].1 += t_end.elapsed().as_secs_f64() * 1e6;

        let t_commit = Instant::now();
        command_buffer.commit();
        times[9].1 += t_commit.elapsed().as_secs_f64() * 1e6;

        let t_wait = Instant::now();
        command_buffer.wait_until_completed();
        times[10].1 += t_wait.elapsed().as_secs_f64() * 1e6;

        let t_read = Instant::now();
        let ptr = output_buffer.contents() as *const f32;
        unsafe {
            for i in 0..m {
                std::ptr::read_volatile(ptr.add(i));
            }
        }
        times[11].1 += t_read.elapsed().as_secs_f64() * 1e6;
    }

    println!("\nStep-by-step timing (avg over {} iterations):", iterations);
    let mut total = 0.0;
    for (name, time) in times {
        let avg = time / iterations as f64;
        println!("  {:20}: {:7.3} us", name, avg);
        total += avg;
    }
    println!("  {:20}: {:7.3} us", "TOTAL", total);
    println!("\nGFLOPS: {:.2}", (m * k * 2) as f64 / (total * 1e-6) / 1e9);
}

fn run_mv(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    weights_buffer: &Buffer,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
    args: &[u8],
    m: usize,
) -> f64 {
    let start = Instant::now();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    // Template kernel signature: (args, src0, src1, dst)
    encoder.set_bytes(
        0,
        args.len() as u64,
        args.as_ptr() as *const std::ffi::c_void,
    );
    encoder.set_buffer(1, Some(weights_buffer), 0);
    encoder.set_buffer(2, Some(input_buffer), 0);
    encoder.set_buffer(3, Some(output_buffer), 0);
    // Q4_K: N_R0 = 2, NSG = 2, ROWS_PER_THREADGROUP = 4
    let nsg = 2u64;
    let nr0 = 2u64;
    let rows_per_tg = nsg * nr0;
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (m as u64 + rows_per_tg - 1) / rows_per_tg,
            height: 1,
            depth: 1,
        },
        MTLSize { width: 32, height: nsg, depth: 1 },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    start.elapsed().as_secs_f64() * 1e6
}

fn generate_q4_k_weights(
    m: usize,
    k: usize,
) -> Vec<infer_train::quant::types::BlockQ4K> {
    use half::f16;
    use infer_train::quant::types::BlockQ4K;

    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i + j) % 64) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}
