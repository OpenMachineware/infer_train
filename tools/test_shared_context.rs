// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Share MetalContext across multiple InferenceContexts to avoid GPU state corruption

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        // Create ONE MetalContext that will be shared
        println!("=== Creating shared MetalContext ===");
        let shared_gpu_ctx = MetalContext::new().unwrap();
        let device = shared_gpu_ctx.device();
        let queue = shared_gpu_ctx.queue();

        // Run 1: Load model with shared MetalContext
        println!("\n=== Run 1: First InferenceContext ===");
        let ctx1 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            shared_gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx1.model.config.clone();

        let mut kv_cache1 = KVCacheGPU::new(device, &config, 2048);
        let reuse_pool1 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let embedding1 = ctx1.model.embedding_lookup_impl(1).unwrap();
        let emb_buffer1 = std::rc::Rc::new(device.new_buffer_with_data(
            embedding1.as_ptr() as *const std::ffi::c_void,
            (embedding1.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let mut forward_pool1 =
            ForwardBufferPool::new(device.clone(), reuse_pool1.clone());
        let layer0 = &ctx1.model.layers[0];
        let cb1 = queue.new_command_buffer();
        let enc1 = cb1.new_compute_command_encoder();

        let output1 = ctx1
            .model
            .forward_layer_gpu_encode(
                layer0,
                0, // layer_idx = 0
                &emb_buffer1,
                &mut kv_cache1,
                &shared_gpu_ctx,
                &mut forward_pool1,
                enc1,
            )
            .unwrap();

        enc1.end_encoding();
        cb1.commit();
        cb1.wait_until_completed();

        let mut h1 = vec![0.0f32; 100];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output1.contents() as *const f32,
                h1.as_mut_ptr(),
                100,
            );
        }
        let nan1 = h1.iter().filter(|v| v.is_nan()).count();
        println!("Run 1 Layer 0 output: first 5 = {:?}", &h1[..5]);
        println!("Run 1 NaN count: {}", nan1);

        // Drop InferenceContext 1 but keep the shared MetalContext
        drop(ctx1);
        println!("\nDropped InferenceContext 1, keeping shared MetalContext");

        // Run 2: Load NEW model with SAME MetalContext
        println!(
            "\n=== Run 2: Second InferenceContext (same MetalContext) ==="
        );
        let ctx2 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            shared_gpu_ctx.clone(),
        )
        .unwrap();

        let mut kv_cache2 = KVCacheGPU::new(device, &config, 2048);
        let reuse_pool2 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let embedding2 = ctx2.model.embedding_lookup_impl(1).unwrap();
        let emb_buffer2 = std::rc::Rc::new(device.new_buffer_with_data(
            embedding2.as_ptr() as *const std::ffi::c_void,
            (embedding2.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let mut forward_pool2 =
            ForwardBufferPool::new(device.clone(), reuse_pool2.clone());
        let layer0 = &ctx2.model.layers[0];
        let cb2 = queue.new_command_buffer();
        let enc2 = cb2.new_compute_command_encoder();

        let output2 = ctx2
            .model
            .forward_layer_gpu_encode(
                layer0,
                0, // layer_idx = 0
                &emb_buffer2,
                &mut kv_cache2,
                &shared_gpu_ctx,
                &mut forward_pool2,
                enc2,
            )
            .unwrap();

        enc2.end_encoding();
        cb2.commit();
        cb2.wait_until_completed();

        let mut h2 = vec![0.0f32; 100];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output2.contents() as *const f32,
                h2.as_mut_ptr(),
                100,
            );
        }
        let nan2 = h2.iter().filter(|v| v.is_nan()).count();
        println!("Run 2 Layer 0 output: first 5 = {:?}", &h2[..5]);
        println!("Run 2 NaN count: {}", nan2);

        println!("\n=== Summary ===");
        if nan1 == 0 && nan2 == 0 {
            println!(
                "✓ Shared MetalContext works for multiple InferenceContexts!"
            );
            println!(
                "Solution confirmed: Use from_gguf_with_gpu_ctx() to share MetalContext"
            );
        } else if nan1 == 0 && nan2 > 0 {
            println!("✗ Shared MetalContext still has issues!");
            println!("Need further investigation");
        } else {
            println!("Unexpected: Run 1 has NaN");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
