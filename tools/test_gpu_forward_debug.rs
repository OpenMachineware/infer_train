// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug GPU forward by testing layer 0 step by step

use infer_train::inference::{Device, InferenceContext};
use metal::MTLResourceOptions;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    println!("Model loaded");
    let model = &ctx.model;
    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    let layer = &model.layers[0];
    let hidden_size = model.config.hidden_size;

    // Test input
    let hidden: Vec<f32> = vec![0.1; hidden_size];
    let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
        hidden.as_ptr() as *const std::ffi::c_void,
        (hidden_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    println!("\n=== Test 1: RMS Norm ===");
    let norm = layer.attention_norm.as_ref().expect("No norm");
    let norm_buffer = gpu_ctx.get_or_create_buffer(norm);
    let normed_buffer = std::rc::Rc::new(device.new_buffer(
        (hidden_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    gpu_ctx.rms_norm_encode(
        enc,
        &hidden_buffer,
        &norm_buffer,
        &normed_buffer,
        1,
        hidden_size,
        1e-5,
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let mut normed = vec![0.0f32; hidden_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            normed_buffer.contents() as *const f32,
            normed.as_mut_ptr(),
            hidden_size,
        );
    }
    println!(
        "Norm: {} values, {} non-zero",
        normed.len(),
        normed.iter().filter(|&&v| v != 0.0).count()
    );

    println!("\n=== Test 2: Q Projection ===");
    let wq = layer.wq.as_ref().expect("No wq");
    let q_out_size = wq.shape[1];
    let q_buffer = std::rc::Rc::new(device.new_buffer(
        (q_out_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    let result = model.gemv_encode(
        enc,
        &normed_buffer,
        wq,
        q_out_size,
        gpu_ctx,
        &q_buffer,
    );
    println!("GEMV result: {:?}", result);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let mut q = vec![0.0f32; q_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            q_buffer.contents() as *const f32,
            q.as_mut_ptr(),
            q_out_size,
        );
    }
    println!(
        "Q: {} values, {} non-zero",
        q.len(),
        q.iter().filter(|&&v| v != 0.0).count()
    );

    println!("\n=== Test 3: Full Layer Forward ===");
    // Now test the actual forward_layer_gpu_encode function
    let mut gpu_kv = ctx.gpu_kv_cache.as_mut().unwrap();

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    let result_buffer = model.forward_layer_gpu_encode(
        layer,
        0,
        &hidden_buffer,
        gpu_kv,
        gpu_ctx,
        device,
        enc,
    );

    println!("Layer forward result: {:?}", result_buffer.is_some());

    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if let Some(buf) = result_buffer {
        let mut output = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents() as *const f32,
                output.as_mut_ptr(),
                hidden_size,
            );
        }
        println!(
            "Output: {} values, {} non-zero",
            output.len(),
            output.iter().filter(|&&v| v != 0.0).count()
        );
        println!("First 5: {:?}", &output[..5]);
    }

    println!("\n=== Done ===");
}
