// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Trace NaN through QKV projections

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLResourceOptions;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();

        println!("=== Creating InferenceContext ===");
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(model_path, gpu_ctx.clone()).unwrap();
        let config = ctx.model.config.clone();
        let hidden_size = config.hidden_size;

        for test_num in 1..=2 {
            println!("\n=== Test {} ===", test_num);

            // Create embedding
            let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
            let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                embedding.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            // RMS norm
            let queue = gpu_ctx.queue();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer = &ctx.model.layers[0];
            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);

            let normed_buffer = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            gpu_ctx.rms_norm_encode(
                encoder,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Q projection
            let wq = layer.wq.as_ref().unwrap();
            let q_out_size = wq.shape[1];
            println!("WQ shape: {:?}, dtype: {:?}", wq.shape, wq.dtype);

            let q_buffer = std::rc::Rc::new(device.new_buffer(
                (q_out_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let q_result = ctx.model.gemv_encode(
                &queue.new_command_buffer().new_compute_command_encoder(),
                &normed_buffer,
                wq,
                q_out_size,
                &gpu_ctx,
                &q_buffer,
            );

            // Commit and wait
            let cb = queue.new_command_buffer();
            // Actually need to use the encoder from above... let me just run gemv directly

            // Actually let's test the GEMV operation directly
            let result = gpu_ctx.mv_q4_0(
                q_out_size,
                hidden_size,
                wq.bytes(),
                &embedding,
            );

            match result {
                Ok(r) => {
                    let nan_count = r.iter().filter(|&&v| v.is_nan()).count();
                    println!("Q projection result: {} NaN, first 5: {:?}", nan_count, &r[..5.min(r.len())]);
                }
                Err(e) => {
                    println!("Q projection error: {}", e);
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
