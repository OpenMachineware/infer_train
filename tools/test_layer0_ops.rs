// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test each operation in layer 0 separately

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::rc::Rc;
use std::cell::RefCell;

fn check_buffer_nan(name: &str, buffer: &metal::BufferRef, size: usize) -> usize {
    let mut data = vec![0.0f32; size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents() as *const f32,
            data.as_mut_ptr(),
            size,
        );
    }
    let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
    if nan_count > 0 {
        println!("{}: {} NaN", name, nan_count);
    }
    nan_count
}

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let config = &model.config;
        let hidden_size = config.hidden_size;
        let layer = &model.layers[0];

        // Create fresh KV cache
        let mut kv = KVCacheGPU::new(device, config, 2048);

        // Get embedding
        let emb = model.embedding_lookup_impl(1).unwrap();
        let emb_buf = Rc::new(device.new_buffer_with_data(
            emb.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let reuse_pool = Rc::new(RefCell::new(Vec::new()));
        let mut pool = infer_train::buffer::ForwardBufferPool::new(device.clone(), reuse_pool);

        // Run attention part
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        let hidden_buffer = pool.register(emb_buf.clone());

        // 1. Attention norm
        let attention_norm = layer.attention_norm.as_ref().unwrap();
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
        let normed_buffer = pool.get_zeroed(hidden_size * 4);
        gpu_ctx.rms_norm_encode(enc, &hidden_buffer, &norm_weight_buffer, &normed_buffer, 1, hidden_size, 1e-5);

        // 2. QKV projections
        let wq = layer.wq.as_ref().unwrap();
        let q_out_size = wq.shape[1];
        let q_buffer = pool.get_zeroed(q_out_size * 4);
        model.gemv_encode(enc, &normed_buffer, wq, q_out_size, gpu_ctx, &q_buffer);

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        println!("=== After attention norm + Q projection ===");
        check_buffer_nan("normed", &normed_buffer, hidden_size);
        check_buffer_nan("Q", &q_buffer, q_out_size);

        // Continue with K, V, etc...
        println!("\nRunning full layer 0...");
        kv.reset_seq_len(); // Reset for second run
        let out = model.forward_layer_gpu_encode(&layer, 0, &emb_buf, &mut kv, gpu_ctx, &mut pool, queue.new_command_buffer().new_compute_command_encoder()).unwrap();

        // Wait and check
        let cb = queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();

        let mut h = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(out.contents() as *const f32, h.as_mut_ptr(), hidden_size);
        }

        let nan_count = h.iter().filter(|&&v| v.is_nan()).count();
        println!("Final output: {} NaN", nan_count);
        if nan_count > 0 {
            println!("NaN positions:");
            for (i, &v) in h.iter().enumerate() {
                if v.is_nan() {
                    println!("  Position {}: NaN", i);
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
