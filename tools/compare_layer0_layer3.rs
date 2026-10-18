// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare: Layer 0 vs Layer 3 kernel by kernel

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    let tokens = tokenizer.encode("Hello");

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            let hidden =
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Process layers 0, 1, 2 first
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let mut layer_input = hidden_buffer.clone();

            for layer_idx in 0..3 {
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let layer_output = model
                    .forward_layer_gpu_encode(
                        &model.layers[layer_idx],
                        layer_idx,
                        &layer_input,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect(&format!("Layer {} failed", layer_idx));
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                layer_input = layer_output;
                kv_cache.advance();
            }

            println!("\n=== Compare Layer 0 vs Layer 3 ===\n");

            // Compare layer 0 and layer 3
            for test_layer_idx in [0, 3] {
                println!("--- Layer {} ---", test_layer_idx);

                let layer = &model.layers[test_layer_idx];
                let pos = kv_cache.seq_len();

                let mut local_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());

                // RMS Norm
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let norm_weight = layer.attention_norm.as_ref().unwrap();
                let norm_weight_buf = gpu_ctx.get_or_create_buffer(norm_weight);
                let normed_buf = local_pool.get_zeroed(hidden_size * 4);

                gpu_ctx.rms_norm_encode(
                    encoder,
                    &layer_input,
                    &norm_weight_buf,
                    &normed_buf,
                    1,
                    hidden_size,
                    1e-5,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                println!("  RMS Norm: {:?}", start.elapsed());

                // Q Projection
                let wq = layer.wq.as_ref().unwrap();
                let q_out_size = wq.shape[1];

                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let q_buf = local_pool.get_zeroed(q_out_size * 4);

                model
                    .gemv_encode(
                        encoder,
                        &normed_buf,
                        wq,
                        q_out_size,
                        gpu_ctx,
                        &q_buf,
                    )
                    .unwrap();
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                println!("  Q Projection: {:?}", start.elapsed());

                // K Projection
                let wk = layer.wk.as_ref().unwrap();
                let k_out_size = wk.shape[1];

                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let k_buf = local_pool.get_zeroed(k_out_size * 4);

                model
                    .gemv_encode(
                        encoder,
                        &normed_buf,
                        wk,
                        k_out_size,
                        gpu_ctx,
                        &k_buf,
                    )
                    .unwrap();
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                println!("  K Projection: {:?}", start.elapsed());

                // V Projection
                let wv = layer.wv.as_ref().unwrap();
                let v_out_size = wv.shape[1];

                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let v_buf = local_pool.get_zeroed(v_out_size * 4);

                model
                    .gemv_encode(
                        encoder,
                        &normed_buf,
                        wv,
                        v_out_size,
                        gpu_ctx,
                        &v_buf,
                    )
                    .unwrap();
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                println!("  V Projection: {:?}", start.elapsed());

                // Flash Attention
                let head_dim = model.config.head_dim;
                let n_heads = q_out_size / head_dim;
                let n_kv_heads = k_out_size / head_dim;

                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

                // Need to update KV cache first
                let k_cache = kv_cache.k_buffer(test_layer_idx);
                let v_cache = kv_cache.v_buffer(test_layer_idx);

                // RoPE
                let q_rope_buf = local_pool.get_zeroed(q_out_size * 4);
                let k_rope_buf = local_pool.get_zeroed(k_out_size * 4);

                gpu_ctx.rope_neox_encode(
                    encoder,
                    &q_buf,
                    &[pos as i32],
                    1,
                    n_heads,
                    head_dim,
                    10000.0,
                    1.0,
                    &q_rope_buf,
                );
                gpu_ctx.rope_neox_encode(
                    encoder,
                    &k_buf,
                    &[pos as i32],
                    1,
                    n_kv_heads,
                    head_dim,
                    10000.0,
                    1.0,
                    &k_rope_buf,
                );

                // KV Update
                gpu_ctx.kv_update_k_encode(
                    encoder,
                    &k_rope_buf,
                    k_cache,
                    pos,
                    kv_cache.max_seq(),
                    n_kv_heads,
                    head_dim,
                );
                gpu_ctx.kv_update_v_encode(
                    encoder,
                    &v_buf,
                    v_cache,
                    pos,
                    kv_cache.max_seq(),
                    n_kv_heads,
                    head_dim,
                );

                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                println!("  RoPE+KV Update: {:?}", start.elapsed());

                // Flash Attention
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let attn_out = local_pool.get_zeroed(q_out_size * 4);

                gpu_ctx.flash_attn_f16_encode(
                    encoder,
                    1,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    pos + 1,
                    kv_cache.max_seq(),
                    &q_rope_buf,
                    k_cache,
                    v_cache,
                    &attn_out,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                println!("  Flash Attention: {:?}", start.elapsed());
            }
        }
    }
}
