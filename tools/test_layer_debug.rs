// Debug layer execution step by step

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(name: &str, buffer: &metal::BufferRef, size: usize) {
    let mut data = vec![0.0f32; size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents() as *const f32,
            data.as_mut_ptr(),
            size,
        );
    }
    let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
    let inf_count = data.iter().filter(|&&v| v.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        println!("{}: {} NaN, {} inf", name, nan_count, inf_count);
    } else {
        // println!("{}: OK (first 3: {:?})", name, &data[..3.min(size)]);
    }
}

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let config = &model.config;
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;

        for run in 1..=2 {
            println!("\n=== RUN {} ===", run);

            let mut kv = KVCacheGPU::new(device, config, 2048);
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            let mut pool = infer_train::buffer::ForwardBufferPool::new(
                device.clone(),
                reuse_pool,
            );

            // Get embedding
            let emb = model.embedding_lookup_impl(1).unwrap();
            let emb_buf = Rc::new(device.new_buffer_with_data(
                emb.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let layer = &model.layers[0];
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            let hidden_buffer = pool.register(emb_buf.clone());
            check_buffer("input", &hidden_buffer, hidden_size);

            // Attention norm
            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(attention_norm);
            let normed_buffer = pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                enc,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );

            // QKV
            let wq = layer.wq.as_ref().unwrap();
            let wk = layer.wk.as_ref().unwrap();
            let wv = layer.wv.as_ref().unwrap();
            let wo = layer.wo.as_ref().unwrap();
            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let v_out_size = wv.shape[1];
            let actual_n_heads = q_out_size / head_dim;
            let actual_n_kv_heads = k_out_size / head_dim;

            let q_buffer = pool.get_zeroed(q_out_size * 4);
            let k_buffer = pool.get_zeroed(k_out_size * 4);
            let v_buffer = pool.get_zeroed(v_out_size * 4);
            model.gemv_encode(
                enc,
                &normed_buffer,
                wq,
                q_out_size,
                gpu_ctx,
                &q_buffer,
            );
            model.gemv_encode(
                enc,
                &normed_buffer,
                wk,
                k_out_size,
                gpu_ctx,
                &k_buffer,
            );
            model.gemv_encode(
                enc,
                &normed_buffer,
                wv,
                v_out_size,
                gpu_ctx,
                &v_buffer,
            );

            // RoPE
            let pos = kv.seq_len();
            let q_rope_buffer = pool.get_zeroed(q_out_size * 4);
            let k_rope_buffer = pool.get_zeroed(k_out_size * 4);
            gpu_ctx.rope_neox_encode(
                enc,
                &q_buffer,
                &[pos as i32],
                1,
                actual_n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope_buffer,
            );
            gpu_ctx.rope_neox_encode(
                enc,
                &k_buffer,
                &[pos as i32],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope_buffer,
            );

            // KV update
            let k_cache = kv.k_buffer(0);
            let v_cache = kv.v_buffer(0);
            gpu_ctx.kv_update_k_encode(
                enc,
                &k_rope_buffer,
                k_cache,
                pos,
                kv.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                enc,
                &v_buffer,
                v_cache,
                pos,
                kv.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            use metal::ResourceRef;
            enc.memory_barrier_with_resources(&[
                k_cache.as_ref() as &ResourceRef,
                v_cache.as_ref() as &ResourceRef,
            ]);

            // Flash attention
            let attn_out_buffer = pool.get_zeroed(q_out_size * 4);
            gpu_ctx.flash_attn_f16_encode(
                enc,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                pos + 1,
                kv.max_seq(),
                &q_rope_buffer,
                k_cache,
                v_cache,
                &attn_out_buffer,
            );

            // Output projection
            let attn_proj_buffer = pool.get_zeroed(hidden_size * 4);
            model.gemv_encode(
                enc,
                &attn_out_buffer,
                wo,
                hidden_size,
                gpu_ctx,
                &attn_proj_buffer,
            );

            // Residual add
            let residual_buffer = pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                enc,
                &hidden_buffer,
                &attn_proj_buffer,
                &residual_buffer,
                hidden_size,
            );

            // FFN norm
            let ffn_norm = layer.ffn_norm.as_ref().unwrap();
            let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
            let ffn_normed_buffer = pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                enc,
                &residual_buffer,
                &ffn_norm_buffer,
                &ffn_normed_buffer,
                1,
                hidden_size,
                1e-5,
            );

            // FFN
            let w1 = layer.w1.as_ref().unwrap();
            let w2 = layer.w2.as_ref().unwrap();
            let w3 = layer.w3.as_ref().unwrap();
            let intermediate_size = w1.shape[1];
            let gate_buffer = pool.get_zeroed(intermediate_size * 4);
            let up_buffer = pool.get_zeroed(intermediate_size * 4);
            let silu_buffer = pool.get_zeroed(intermediate_size * 4);
            let hidden_ffn_buffer = pool.get_zeroed(intermediate_size * 4);
            let ffn_out_buffer = pool.get_zeroed(hidden_size * 4);

            model.gemv_encode(
                enc,
                &ffn_normed_buffer,
                w1,
                intermediate_size,
                gpu_ctx,
                &gate_buffer,
            );
            model.gemv_encode(
                enc,
                &ffn_normed_buffer,
                w3,
                intermediate_size,
                gpu_ctx,
                &up_buffer,
            );
            gpu_ctx.silu_encode(
                enc,
                &gate_buffer,
                &silu_buffer,
                intermediate_size,
            );
            gpu_ctx.mul_encode(
                enc,
                &silu_buffer,
                &up_buffer,
                &hidden_ffn_buffer,
                intermediate_size,
            );
            model.gemv_encode(
                enc,
                &hidden_ffn_buffer,
                w2,
                hidden_size,
                gpu_ctx,
                &ffn_out_buffer,
            );

            // Final residual
            let final_buffer = pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                enc,
                &residual_buffer,
                &ffn_out_buffer,
                &final_buffer,
                hidden_size,
            );

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Check all buffers
            println!("\nChecking buffers:");
            check_buffer("normed", &normed_buffer, hidden_size);
            check_buffer("Q", &q_buffer, q_out_size);
            check_buffer("K", &k_buffer, k_out_size);
            check_buffer("V", &v_buffer, v_out_size);
            check_buffer("Q_rope", &q_rope_buffer, q_out_size);
            check_buffer("K_rope", &k_rope_buffer, k_out_size);
            check_buffer("attn_out", &attn_out_buffer, q_out_size);
            check_buffer("attn_proj", &attn_proj_buffer, hidden_size);
            check_buffer("residual", &residual_buffer, hidden_size);
            check_buffer("ffn_normed", &ffn_normed_buffer, hidden_size);
            check_buffer("gate", &gate_buffer, intermediate_size);
            check_buffer("up", &up_buffer, intermediate_size);
            check_buffer("silu", &silu_buffer, intermediate_size);
            check_buffer("hidden_ffn", &hidden_ffn_buffer, intermediate_size);
            check_buffer("ffn_out", &ffn_out_buffer, hidden_size);
            check_buffer("FINAL", &final_buffer, hidden_size);

            kv.advance();
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
