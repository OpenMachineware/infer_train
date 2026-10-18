// Test if attention produces NaN

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::cell::RefCell;
use std::rc::Rc;

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
        let layer = &model.layers[0];

        // Create TWO KV caches
        let mut kv1 = KVCacheGPU::new(device, config, 2048);
        let mut kv2 = KVCacheGPU::new(device, config, 2048);

        // Get embedding
        let emb = model.embedding_lookup_impl(1).unwrap();
        let emb_buf1 = Rc::new(device.new_buffer_with_data(
            emb.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        let emb_buf2 = Rc::new(device.new_buffer_with_data(
            emb.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let reuse_pool1 = Rc::new(RefCell::new(Vec::new()));
        let mut pool1 = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool1,
        );
        let reuse_pool2 = Rc::new(RefCell::new(Vec::new()));
        let mut pool2 = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool2,
        );

        // Run attention part twice
        for run in 1..=2 {
            println!("\n=== Run {} ===", run);

            let kv = if run == 1 { &mut kv1 } else { &mut kv2 };
            let pool = if run == 1 { &mut pool1 } else { &mut pool2 };
            let emb_buf = if run == 1 { &emb_buf1 } else { &emb_buf2 };

            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            let hidden_buffer = pool.register(emb_buf.clone());

            // 1. Attention norm
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

            // 2. QKV projections
            let wq = layer.wq.as_ref().unwrap();
            let wk = layer.wk.as_ref().unwrap();
            let wv = layer.wv.as_ref().unwrap();
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

            // 3. RoPE
            let q_rope_buffer = pool.get_zeroed(q_out_size * 4);
            let k_rope_buffer = pool.get_zeroed(k_out_size * 4);
            let pos = kv.seq_len();

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

            // 4. KV cache update
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

            // 5. Flash attention
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

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Check for NaN
            let mut attn_out = vec![0.0f32; q_out_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    attn_out_buffer.contents() as *const f32,
                    attn_out.as_mut_ptr(),
                    q_out_size,
                );
            }

            let nan_count = attn_out.iter().filter(|&&v| v.is_nan()).count();
            println!("Attention output: {} NaN", nan_count);
            if nan_count > 0 {
                println!("NaN positions:");
                for (i, &v) in attn_out.iter().enumerate() {
                    if v.is_nan() {
                        println!("  Position {}: NaN", i);
                        if i > 20 {
                            println!("  ... (showing first 20)");
                            break;
                        }
                    }
                }
            }
            println!("First 5 attention: {:?}", &attn_out[..5]);

            kv.advance();
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
