// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile each operator in decode to find bottlenecks

use infer_train::model::llama::LlamaModel;
use metal::{Device, MTLResourceOptions};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Profiling: {}", model_path);

    // Load model
    let model =
        LlamaModel::from_gguf(model_path).expect("Failed to load model");
    println!(
        "Model loaded: hidden={}, layers={}",
        model.config.hidden_size, model.config.n_layers
    );

    // Create Metal device
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test each layer
    let hidden_size = model.config.hidden_size;
    let intermediate_size = model.layers[0]
        .w1
        .as_ref()
        .map(|w| w.shape[1])
        .unwrap_or(hidden_size * 4);

    // Create test input buffer
    let input_data = vec![1.0f32; hidden_size];
    let input_buffer = Rc::new(device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (hidden_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    // Profile each layer
    for (layer_idx, layer) in model.layers.iter().enumerate().take(3) {
        println!("\n=== Layer {} ===", layer_idx);

        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        // Start timing
        let start = std::time::Instant::now();

        // Profile gate GEMV
        if let Some(w1) = &layer.w1 {
            let gate_weight_buf = device.new_buffer_with_data(
                w1.bytes().as_ptr() as *const std::ffi::c_void,
                w1.bytes().len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let gate_output = Rc::new(device.new_buffer(
                (intermediate_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let t0 = std::time::Instant::now();
            // Would call GEMV here
            let gemv_time = t0.elapsed();
            println!("  Gate GEMV: {:?}", gemv_time);
        }

        // Profile up GEMV
        if let Some(w3) = &layer.w3 {
            let up_weight_buf = device.new_buffer_with_data(
                w3.bytes().as_ptr() as *const std::ffi::c_void,
                w3.bytes().len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let up_output = Rc::new(device.new_buffer(
                (intermediate_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let t0 = std::time::Instant::now();
            // Would call GEMV here
            let gemv_time = t0.elapsed();
            println!("  Up GEMV: {:?}", gemv_time);
        }

        // Profile down GEMV
        if let Some(w2) = &layer.w2 {
            let down_weight_buf = device.new_buffer_with_data(
                w2.bytes().as_ptr() as *const std::ffi::c_void,
                w2.bytes().len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let down_output = Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let t0 = std::time::Instant::now();
            // Would call GEMV here
            let gemv_time = t0.elapsed();
            println!("  Down GEMV: {:?}", gemv_time);
        }

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let layer_time = start.elapsed();
        println!("  Total layer time: {:?}", layer_time);
    }
}
