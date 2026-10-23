// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Command buffer scheduling test
//!
//! llama.cpp uses command_buffer.enqueue() before commit.
//! This tests if enqueue affects decode performance.

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLResourceOptions;
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::time::Instant;

        // Load model
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Create KV cache
        let mut kv_cache = KVCacheGPU::new(
            &device,
            &ctx.config,
            ctx.config.max_position_embeddings,
        );

        // Warmup
        let warmup_tokens: Vec<u32> = (0..10).collect();
        ctx.model.forward_gpu_native_batched(
            &warmup_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        // Reset KV cache for benchmark
        kv_cache.clear();

        // Prefill with longer context to establish KV cache
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        // Get model dimensions
        let hidden_size = ctx.config.hidden_size;

        println!("\n=== Testing Command Buffer Scheduling ===");
        println!("Hidden size: {}", hidden_size);

        // Create test buffers
        let input_buffer: Rc<metal::Buffer> =
            Rc::new(device.new_buffer_with_data(
                vec![1.0f32; hidden_size].as_ptr() as *const _,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

        // Get weights from layer 0
        let layer = &ctx.model.layers[0];
        let wq = layer.wq.as_ref().expect("No wq");
        let wk = layer.wk.as_ref().expect("No wk");
        let wv = layer.wv.as_ref().expect("No wv");

        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];

        // Create output buffers
        let reuse_pool: Rc<RefCell<Vec<Rc<metal::Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool.clone(),
        );

        let q_buffer = forward_pool.get(q_out_size * 4);
        let k_buffer = forward_pool.get(k_out_size * 4);
        let v_buffer = forward_pool.get(v_out_size * 4);

        let iterations = 100;

        // Test 1: Without enqueue (baseline)
        println!("\n--- Test 1: Without enqueue() ---");
        let start = Instant::now();

        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wq,
                q_out_size,
                &gpu_ctx,
                &q_buffer,
            );
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wk,
                k_out_size,
                &gpu_ctx,
                &k_buffer,
            );
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wv,
                v_out_size,
                &gpu_ctx,
                &v_buffer,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let no_enqueue_time = start.elapsed();
        let no_enqueue_avg_us =
            no_enqueue_time.as_micros() as f64 / iterations as f64;
        println!("Total: {:?}", no_enqueue_time);
        println!("Avg per iteration: {:.2} µs", no_enqueue_avg_us);

        // Test 2: With enqueue (llama.cpp style)
        println!("\n--- Test 2: With enqueue() (llama.cpp style) ---");
        let start = Instant::now();

        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            cb.enqueue(); // Pre-enqueue (llama.cpp does this on line 565)

            let encoder = cb.new_compute_command_encoder();

            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wq,
                q_out_size,
                &gpu_ctx,
                &q_buffer,
            );
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wk,
                k_out_size,
                &gpu_ctx,
                &k_buffer,
            );
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wv,
                v_out_size,
                &gpu_ctx,
                &v_buffer,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let enqueue_time = start.elapsed();
        let enqueue_avg_us =
            enqueue_time.as_micros() as f64 / iterations as f64;
        println!("Total: {:?}", enqueue_time);
        println!("Avg per iteration: {:.2} µs", enqueue_avg_us);

        let speedup = no_enqueue_avg_us / enqueue_avg_us;
        println!("\n=== Results ===");
        println!("Speedup: {:.2}x", speedup);

        if speedup > 1.05 {
            println!("✅ enqueue() improves performance!");
        } else if speedup > 0.95 {
            println!("⚠️ No significant difference");
        } else {
            println!("❌ enqueue() is slower");
        }

        // Test 3: Check if our DAG executor uses enqueue
        println!("\n--- Checking DAG executor implementation ---");
        let dag_uses_enqueue = true; // From src/dag/executor.rs line 206
        println!("DAG executor uses enqueue(): {}", dag_uses_enqueue);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!(
            "This benchmark requires Metal support (macOS with metal feature)"
        );
    }
}
