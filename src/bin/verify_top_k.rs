// Verify TOP_K kernel correctness
use metal::{Device, MTLSize};

// Must match llama.cpp struct layout exactly
#[repr(C)]
struct TopKArgs {
    ne00: i32, // number of columns (elements per row)
    ne01: i32, // rows
    ne02: i32,
    ne03: i32,
    nb01: u64, // row stride in src0
    nb02: u64,
    nb03: u64,
    top_k: i32, // k
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let source = include_str!("../../shaders/top_k.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Failed to compile");
    let kernel =
        library.get_function("top_k_f32_i32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    // Test: find top-5 from 20 elements
    let n: i32 = 20;
    let k: i32 = 5;
    let input_data: Vec<f32> = vec![
        3.0, 1.0, 5.0, 2.0, 4.0, 10.0, 8.0, 6.0, 7.0, 9.0, 15.0, 11.0, 13.0,
        12.0, 14.0, 20.0, 18.0, 16.0, 17.0, 19.0,
    ];

    // CPU reference: find top-k indices (descending order)
    let mut indexed: Vec<(usize, f32)> =
        input_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let cpu_topk: Vec<i32> =
        indexed.iter().take(k as usize).map(|(i, _)| *i as i32).collect();

    // GPU execution
    let input_buf = device.new_buffer(
        (n as usize * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (k as usize * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input_data.as_ptr(),
            input_buf.contents() as *mut f32,
            n as usize,
        );
    }

    // Args struct
    let args = TopKArgs {
        ne00: n,
        ne01: 1,
        ne02: 1,
        ne03: 1,
        nb01: n as u64 * 4,
        nb02: n as u64 * 4,
        nb03: n as u64 * 4,
        top_k: k,
    };

    // Threadgroup memory for histogram (256 * 4 bytes)
    let tg_mem = 256 * 4;
    let tg_mem_4 = 4; // for sh_bucket, sh_above, out_count

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<TopKArgs>() as u64,
        &args as *const TopKArgs as *const std::ffi::c_void,
    );
    enc.set_buffer(1, Some(&input_buf), 0);
    enc.set_buffer(2, Some(&output_buf), 0);
    enc.set_threadgroup_memory_length(0, tg_mem as u64);
    enc.set_threadgroup_memory_length(1, tg_mem_4 as u64);
    enc.set_threadgroup_memory_length(2, tg_mem_4 as u64);
    enc.set_threadgroup_memory_length(3, tg_mem_4 as u64);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_out = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const i32,
            k as usize,
        )
    };

    println!("=== TOP_K ===");
    println!("Input: {} values, find top-{}", n, k);
    println!(
        "Expected values: {:?}",
        cpu_topk.iter().map(|&i| input_data[i as usize]).collect::<Vec<_>>()
    );
    println!("Got indices: {:?}", gpu_out);
    println!(
        "Got values: {:?}",
        gpu_out.iter().map(|&i| input_data[i as usize]).collect::<Vec<_>>()
    );

    // Check if the VALUES match (order may differ)
    let mut gpu_values: Vec<f32> =
        gpu_out.iter().map(|&i| input_data[i as usize]).collect();
    let mut cpu_values: Vec<f32> =
        cpu_topk.iter().map(|&i| input_data[i as usize]).collect();
    gpu_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
    cpu_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let correct = gpu_values == cpu_values;

    if correct {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
