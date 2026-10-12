// Verify ARGSORT kernel correctness
use metal::{Device, MTLSize};

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let source = include_str!("../../shaders/argsort.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Failed to compile");
    let kernel =
        library.get_function("argsort_f32_i32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    // Test: sort 16 elements, get top-8
    let n: i32 = 16;
    let topk: i32 = 8;
    let input_data: Vec<f32> = vec![
        15.0, 3.0, 11.0, 7.0, 13.0, 1.0, 9.0, 5.0, 14.0, 2.0, 10.0, 6.0, 12.0,
        0.0, 8.0, 4.0,
    ];

    // CPU reference: descending sort
    let mut indexed: Vec<(usize, f32)> =
        input_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let cpu_indices: Vec<i32> =
        indexed.iter().take(topk as usize).map(|(i, _)| *i as i32).collect();

    // GPU execution
    let input_buf = device.new_buffer(
        (n as usize * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (topk as usize * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input_data.as_ptr(),
            input_buf.contents() as *mut f32,
            n as usize,
        );
    }

    // Threadgroup memory for bitonic sort
    let sort_order = 1i32; // Descending
    let tg_mem = n as u64 * 4;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &topk as *const i32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &sort_order as *const i32 as *const std::ffi::c_void);
    enc.set_threadgroup_memory_length(0, tg_mem as u64);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: n as u64, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_out = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const i32,
            topk as usize,
        )
    };

    println!("=== ARGSORT (Descending) ===");
    println!("Input: {} values, get top-{}", n, topk);
    println!("Expected: {:?}", cpu_indices);
    println!("Got:      {:?}", gpu_out);

    let mut correct = true;
    for i in 0..topk as usize {
        if i < gpu_out.len() && gpu_out[i] == cpu_indices[i] {
            // OK
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
