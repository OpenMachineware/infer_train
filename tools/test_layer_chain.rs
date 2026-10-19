// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test multi-layer execution WITHOUT buffer reuse

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();

        let mut kv_cache = KVCacheGPU::new(device, &config, 2048);

        // DO NOT use reuse pool - each layer creates fresh buffers
        // This tests if buffer reuse is the problem
        let reuse_pool = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
        let emb_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            embedding.as_ptr() as *const std::ffi::c_void,
            (embedding.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        println!("=== Multi-layer test WITHOUT buffer reuse ===");
        println!("Embedding: first 5 = {:?}", &embedding[..5]);

        let mut hidden = emb_buffer;
        let num_layers = ctx.model.layers.len();

        // Test only first 10 layers to save time
        for layer_idx in 0..10.min(num_layers) {
            // Create a new pool that will DROP all buffers after use (no reuse)
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let layer = &ctx.model.layers[layer_idx];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            let output = ctx
                .model
                .forward_layer_gpu_encode(
                    layer,
                    layer_idx,
                    &hidden,
                    &mut kv_cache,
                    &gpu_ctx,
                    &mut forward_pool,
                    enc,
                )
                .unwrap();

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Check output
            let mut h = vec![0.0f32; 100];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output.contents() as *const f32,
                    h.as_mut_ptr(),
                    100,
                );
            }
            let nan_count = h.iter().filter(|v| v.is_nan()).count();
            let zero_count = h.iter().filter(|v| **v == 0.0).count();

            println!(
                "Layer {} output: {} NaN, {} zero, first 5 = {:?}",
                layer_idx,
                nan_count,
                zero_count,
                &h[..5]
            );

            if nan_count > 0 {
                println!("\n*** NaN detected at Layer {}! ***", layer_idx);
                return;
            }

            // Drop forward_pool to release buffers
            // But keep the output buffer via Rc
            drop(forward_pool);
            hidden = output;

            // Check reuse pool
            println!("  Reuse pool size: {}", reuse_pool.borrow().len());
        }

        println!("\n=== Test completed ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
