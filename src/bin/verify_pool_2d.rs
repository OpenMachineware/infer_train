// Verify POOL_2D kernel correctness vs CPU reference
use metal::{Device, MTLSize};

#[repr(C)]
#[derive(Clone, Copy)]
struct Pool2DArgs {
    k0: i32,
    k1: i32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    IH: i64,
    IW: i64,
    OH: i64,
    OW: i64,
    np: i64,
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Compile kernel
    let source = include_str!("../../shaders/pool_2d.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Failed to compile");
    let max_kernel =
        library.get_function("pool_2d_max_f32", None).expect("No max kernel");
    let avg_kernel =
        library.get_function("pool_2d_avg_f32", None).expect("No avg kernel");
    let max_pipeline = device
        .new_compute_pipeline_state_with_function(&max_kernel)
        .expect("No pipeline");
    let avg_pipeline = device
        .new_compute_pipeline_state_with_function(&avg_kernel)
        .expect("No pipeline");

    // Test: 8x8 input, 2x2 kernel, stride 2, no padding → 4x4 output
    // Note: IC is computed as np / (OH * OW) in the kernel
    let args = Pool2DArgs {
        k0: 2,
        k1: 2,
        s0: 2,
        s1: 2,
        p0: 0,
        p1: 0,
        IH: 8,
        IW: 8,
        OH: 4,
        OW: 4,
        np: 16,
    };

    let input_size = (args.IH * args.IW) as usize;
    let output_size = (args.OH * args.OW) as usize;
    let input_data: Vec<f32> =
        (0..input_size as i32).map(|i| i as f32).collect();

    // CPU reference: max pooling
    let cpu_max: Vec<f32> = (0..output_size as i32)
        .map(|idx| {
            let oh = idx / args.OW as i32;
            let ow = idx % args.OW as i32;
            let mut max_val = f32::NEG_INFINITY;
            for ky in 0..args.k1 {
                let in_y = oh * args.s1 + ky;
                for kx in 0..args.k0 {
                    let in_x = ow * args.s0 + kx;
                    let val =
                        input_data[(in_y * args.IW as i32 + in_x) as usize];
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            max_val
        })
        .collect();

    // GPU execution
    let input_buf = device.new_buffer(
        (input_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (output_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input_data.as_ptr(),
            input_buf.contents() as *mut f32,
            input_size,
        );
    }

    // Dispatch max pooling
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&max_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<Pool2DArgs>() as u64,
        &args as *const Pool2DArgs as *const std::ffi::c_void,
    );
    enc.set_buffer(1, Some(&input_buf), 0);
    enc.set_buffer(2, Some(&output_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize { width: output_size as u64, height: 1, depth: 1 },
        MTLSize { width: 64, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_out = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const f32,
            output_size,
        )
    };
    let max_diff = cpu_max
        .iter()
        .zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("=== POOL_2D Max Pooling ===");
    println!(
        "Input: {}x{}, Kernel: {}x{}, Stride: {}",
        args.IH, args.IW, args.k0, args.k1, args.s0
    );
    println!("Max diff: {}", max_diff);
    if max_diff < 1e-5 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
        println!("Expected: {:?}", &cpu_max[..8]);
        println!("Got: {:?}", &gpu_out[..8]);
    }

    // CPU reference: avg pooling
    let cpu_avg: Vec<f32> = (0..output_size as i32)
        .map(|idx| {
            let oh = idx / args.OW as i32;
            let ow = idx % args.OW as i32;
            let mut sum = 0.0f32;
            for ky in 0..args.k1 {
                let in_y = oh * args.s1 + ky;
                for kx in 0..args.k0 {
                    let in_x = ow * args.s0 + kx;
                    sum += input_data[(in_y * args.IW as i32 + in_x) as usize];
                }
            }
            sum / (args.k0 * args.k1) as f32
        })
        .collect();

    // Dispatch avg pooling
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&avg_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<Pool2DArgs>() as u64,
        &args as *const Pool2DArgs as *const std::ffi::c_void,
    );
    enc.set_buffer(1, Some(&input_buf), 0);
    enc.set_buffer(2, Some(&output_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize { width: output_size as u64, height: 1, depth: 1 },
        MTLSize { width: 64, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_avg = unsafe {
        std::slice::from_raw_parts(
            output_buf.contents() as *const f32,
            output_size,
        )
    };
    let max_diff = cpu_avg
        .iter()
        .zip(gpu_avg.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\n=== POOL_2D Avg Pooling ===");
    println!("Max diff: {}", max_diff);
    if max_diff < 1e-5 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
