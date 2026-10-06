// Verify TOP_K kernel correctness
use metal::{Device, MTLSize};

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

    // Need threadgroup memory for histogram (256 * 4) + 2 uints
    let tg_mem = 256 * 4 + 8;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &k as *const i32 as *const std::ffi::c_void);
    enc.set_threadgroup_memory_length(0, tg_mem as u64);
    enc.set_threadgroup_memory_length(1, 4);
    enc.set_threadgroup_memory_length(2, 4);
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
    println!("Expected indices: {:?}", cpu_topk);
    println!("Got indices: {:?}", gpu_out);

    // Check if results match (values should be in descending order)
    let mut correct = true;
    for i in 0..k as usize {
        if i < gpu_out.len() {
            let gpu_val = input_data[gpu_out[i] as usize];
            let cpu_val = input_data[cpu_topk[i] as usize];
            if (gpu_val - cpu_val).abs() > 1e-5 {
                correct = false;
            }
        } else {
            correct = false;
        }
    }

    if correct {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
