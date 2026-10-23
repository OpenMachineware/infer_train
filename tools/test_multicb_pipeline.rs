// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Multi-command buffer pipelining test
//!
//! Tests if using multiple command buffers can improve decode bandwidth
//! by allowing GPU to execute kernels from different buffers concurrently.
//!
//! Hypothesis: llama.cpp's multi-CB architecture enables kernel overlap
//! across layers, improving overall bandwidth utilization from 20% to 30%+.

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
        let n_heads = ctx.config.n_heads;
        let n_kv_heads = ctx.config.n_kv_heads;
        let intermediate_size = ctx.config.intermediate_size;

        println!("\n=== Testing Multi-CB Pipelining ===");
        println!("Hidden size: {}", hidden_size);
        println!("Heads: {} (KV: {})", n_heads, n_kv_heads);
        println!("Intermediate: {}", intermediate_size);

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

        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        let wk_buffer = gpu_ctx.get_or_create_buffer(wk);
        let wv_buffer = gpu_ctx.get_or_create_buffer(wv);

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

        // Test 1: Sequential dispatch (single command buffer)
        println!("\n--- Test 1: Sequential (1 CB) ---");
        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            // Q projection
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wq,
                q_out_size,
                &gpu_ctx,
                &q_buffer,
            );
            // K projection
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wk,
                k_out_size,
                &gpu_ctx,
                &k_buffer,
            );
            // V projection
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

        let sequential_time = start.elapsed();
        let sequential_avg_us =
            sequential_time.as_micros() as f64 / iterations as f64;
        println!("Total: {:?}", sequential_time);
        println!("Avg per iteration: {:.2} µs", sequential_avg_us);

        // Test 2: Parallel dispatch (3 command buffers)
        println!("\n--- Test 2: Parallel (3 CBs) ---");
        let start = Instant::now();

        for _ in 0..iterations {
            // Create 3 command buffers
            let cb_q = queue.new_command_buffer_with_unretained_references();
            let cb_k = queue.new_command_buffer_with_unretained_references();
            let cb_v = queue.new_command_buffer_with_unretained_references();

            // Encode Q
            let enc_q = cb_q.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &enc_q,
                &input_buffer,
                hidden_size,
                wq,
                q_out_size,
                &gpu_ctx,
                &q_buffer,
            );
            enc_q.end_encoding();

            // Encode K
            let enc_k = cb_k.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &enc_k,
                &input_buffer,
                hidden_size,
                wk,
                k_out_size,
                &gpu_ctx,
                &k_buffer,
            );
            enc_k.end_encoding();

            // Encode V
            let enc_v = cb_v.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &enc_v,
                &input_buffer,
                hidden_size,
                wv,
                v_out_size,
                &gpu_ctx,
                &v_buffer,
            );
            enc_v.end_encoding();

            // Commit all
            cb_q.commit();
            cb_k.commit();
            cb_v.commit();

            // Wait for all to complete
            cb_q.wait_until_completed();
            cb_k.wait_until_completed();
            cb_v.wait_until_completed();
        }

        let parallel_time = start.elapsed();
        let parallel_avg_us =
            parallel_time.as_micros() as f64 / iterations as f64;
        println!("Total: {:?}", parallel_time);
        println!("Avg per iteration: {:.2} µs", parallel_avg_us);

        // Calculate improvement
        let speedup = sequential_avg_us / parallel_avg_us;
        println!("\n=== Results ===");
        println!("Speedup: {:.2}x", speedup);
        println!(
            "Time saved: {:.2} µs per iteration",
            sequential_avg_us - parallel_avg_us
        );

        if speedup > 1.1 {
            println!("\n✅ Multi-CB improves performance - pipelining works!");
        } else if speedup > 0.95 {
            println!(
                "\n⚠️ No significant improvement - Metal scheduler already optimizes sequential dispatch"
            );
        } else {
            println!(
                "\n❌ Multi-CB is SLOWER - overhead exceeds parallelism benefit"
            );
        }

        // Test 3: Simulate full layer with 5 projections
        println!("\n--- Test 3: Full Layer Sequential (1 CB) ---");
        let gate = layer.w1.as_ref().expect("No gate");
        let up = layer.w3.as_ref().expect("No up");

        let gate_out_size = gate.shape[1];
        let up_out_size = up.shape[1];

        let gate_out = forward_pool.get(gate_out_size * 4);
        let up_out = forward_pool.get(up_out_size * 4);

        let start = Instant::now();

        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            // 5 projections: Q, K, V, gate, up
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
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                gate,
                gate_out_size,
                &gpu_ctx,
                &gate_out,
            );
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                up,
                up_out_size,
                &gpu_ctx,
                &up_out,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let full_sequential_time = start.elapsed();
        let full_sequential_avg_us =
            full_sequential_time.as_micros() as f64 / iterations as f64;
        println!("Total: {:?}", full_sequential_time);
        println!("Avg per iteration: {:.2} µs", full_sequential_avg_us);

        // Test 4: Full layer with 2 CBs (3+2 split)
        println!("\n--- Test 4: Full Layer Parallel (2 CBs) ---");
        let start = Instant::now();

        for _ in 0..iterations {
            let cb1 = queue.new_command_buffer_with_unretained_references();
            let cb2 = queue.new_command_buffer_with_unretained_references();

            // CB1: Q, K, V
            let enc1 = cb1.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &enc1,
                &input_buffer,
                hidden_size,
                wq,
                q_out_size,
                &gpu_ctx,
                &q_buffer,
            );
            ctx.model.gemv_encode(
                &enc1,
                &input_buffer,
                hidden_size,
                wk,
                k_out_size,
                &gpu_ctx,
                &k_buffer,
            );
            ctx.model.gemv_encode(
                &enc1,
                &input_buffer,
                hidden_size,
                wv,
                v_out_size,
                &gpu_ctx,
                &v_buffer,
            );
            enc1.end_encoding();

            // CB2: gate, up
            let enc2 = cb2.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &enc2,
                &input_buffer,
                hidden_size,
                gate,
                gate_out_size,
                &gpu_ctx,
                &gate_out,
            );
            ctx.model.gemv_encode(
                &enc2,
                &input_buffer,
                hidden_size,
                up,
                up_out_size,
                &gpu_ctx,
                &up_out,
            );
            enc2.end_encoding();

            cb1.commit();
            cb2.commit();

            cb1.wait_until_completed();
            cb2.wait_until_completed();
        }

        let full_parallel_time = start.elapsed();
        let full_parallel_avg_us =
            full_parallel_time.as_micros() as f64 / iterations as f64;
        println!("Total: {:?}", full_parallel_time);
        println!("Avg per iteration: {:.2} µs", full_parallel_avg_us);

        let full_speedup = full_sequential_avg_us / full_parallel_avg_us;
        println!("\n=== Full Layer Results ===");
        println!("Speedup: {:.2}x", full_speedup);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!(
            "This benchmark requires Metal support (macOS with metal feature)"
        );
    }
}
