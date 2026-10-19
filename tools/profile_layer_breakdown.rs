// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile detailed breakdown of one layer

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::time::Instant;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();
        let mut kv = KVCacheGPU::new(device.clone(), &config, 2048);

        // Prefill to position 10
        let prefill_tokens: Vec<u32> = (1..=10).collect();
        let _ = ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv,
            &gpu_ctx,
        );

        // Profile one layer at position 10
        if ctx.model.layers.is_empty() {
            eprintln!("No layers!");
            return;
        }

        let layer = &ctx.model.layers[0];
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;

        // Create test input
        let input_data: Vec<f32> = vec![0.1; hidden_size];
        let input_buffer = Rc::new(device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Create buffer pool
        let reuse_pool: Rc<RefCell<Vec<Rc<metal::Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        let queue = gpu_ctx.queue();
        let n_iterations = 20;
        let mut times = std::collections::HashMap::new();

        for iter in 0..n_iterations {
            forward_pool.clear();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Encode layer operations
            let layer_output = ctx.model.forward_layer_gpu_encode(
                &ctx.model.layers[0],
                0,
                &input_buffer,
                &mut kv,
                &gpu_ctx,
                &mut forward_pool,
                encoder,
            );

            encoder.end_encoding();

            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();

            if let Some(layer_output) = layer_output {
                // Don't use the output, just measure time
                let _ = layer_output;
            }
        }

        println!("=== Single Layer Profile ===");
        println!("Layer 0 at position 10");
        println!("Note: This measures the entire layer as one unit");
        println!(
            "      To get detailed breakdown, need to modify forward_layer_gpu_encode"
        );
        println!(
            "\nNeed to add timing inside forward_layer_gpu_encode to see which operation is slow"
        );
    }
}
