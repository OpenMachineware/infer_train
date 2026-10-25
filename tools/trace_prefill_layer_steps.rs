// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Trace prefill layer step by step

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::model::tensor::DataType;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::cell::RefCell;
        use std::rc::Rc;

        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("Model: {}", model_path);

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load GPU model");

        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let mut kv_cache = KVCacheGPU::new(gpu_ctx.device(), &ctx.config, 2048);

        let tokens = vec![1u32, 2u32];
        let batch = tokens.len();
        let hidden_size = ctx.config.hidden_size;

        // Manual embedding lookup
        let mut all_embeddings = vec![0.0f32; batch * hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            let embedding = ctx
                .model
                .embedding_lookup_impl(token)
                .expect("Embedding failed");
            all_embeddings[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&embedding);
        }

        let embed_buffer = Rc::new(gpu_ctx.device().new_buffer_with_data(
            all_embeddings.as_ptr() as *const std::ffi::c_void,
            (batch * hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Create command buffer
        let queue = gpu_ctx.queue();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();

        // Setup forward pool
        let reuse_pool: Rc<RefCell<Vec<Rc<metal::Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool = infer_train::buffer::ForwardBufferPool::new(
            gpu_ctx.device().clone(),
            reuse_pool.clone(),
        );

        let layer = &ctx.model.layers[0];
        let input_buffer = forward_pool.register(embed_buffer.clone());

        let attention_norm =
            layer.attention_norm.as_ref().expect("No attention norm");
        let wq = layer.wq.as_ref().expect("No wq");
        let wk = layer.wk.as_ref().expect("No wk");
        let wv = layer.wv.as_ref().expect("No wv");

        let head_dim = ctx.config.head_dim;
        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];
        let n_heads = q_out_size / head_dim;
        let n_kv_heads = k_out_size / head_dim;

        println!("\n=== Tracing prefill layer 0 ===");
        println!(
            "batch={}, hidden_size={}, n_heads={}, n_kv_heads={}, head_dim={}",
            batch, hidden_size, n_heads, n_kv_heads, head_dim
        );
        println!(
            "q_out={}, k_out={}, v_out={}",
            q_out_size, k_out_size, v_out_size
        );
        println!(
            "wq dtype={:?}, wk dtype={:?}, wv dtype={:?}",
            wq.dtype, wk.dtype, wv.dtype
        );

        // Step 1: RMS Norm
        println!("\n1. RMS Norm");
        let normed_buffer = forward_pool.get(batch * hidden_size * 4);
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
        gpu_ctx.rms_norm_encode(
            encoder,
            &input_buffer,
            &norm_weight_buffer,
            &normed_buffer,
            batch,
            hidden_size,
            ctx.config.rms_norm_eps,
        );

        // Step 2: F32->F16 conversion
        println!("2. F32->F16");
        let normed_f16_buffer = forward_pool.get(batch * hidden_size * 2);
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &normed_buffer,
            &normed_f16_buffer,
            batch * hidden_size,
        );

        // Step 3: Q projection
        println!("3. Q projection (dtype={:?})", wq.dtype);
        let q_buffer = forward_pool.get(batch * q_out_size * 4);
        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);

        match wq.dtype {
            DataType::Q4_K => {
                gpu_ctx.mul_mm_q4_k_encode(
                    encoder,
                    &wq_buffer,
                    &normed_f16_buffer,
                    &q_buffer,
                    q_out_size,
                    batch,
                    hidden_size,
                );
            }
            DataType::Q5_K => {
                gpu_ctx.mul_mm_q5_k_encode(
                    encoder,
                    &wq_buffer,
                    &normed_f16_buffer,
                    &q_buffer,
                    q_out_size,
                    batch,
                    hidden_size,
                );
            }
            DataType::Q6_K => {
                gpu_ctx.mul_mm_q6_k_encode(
                    encoder,
                    &wq_buffer,
                    &normed_f16_buffer,
                    &q_buffer,
                    q_out_size,
                    batch,
                    hidden_size,
                );
            }
            _ => panic!("Unsupported dtype {:?}", wq.dtype),
        }

        // Step 4: K projection
        println!("4. K projection (dtype={:?})", wk.dtype);
        let k_buffer = forward_pool.get(batch * k_out_size * 4);
        let wk_buffer = gpu_ctx.get_or_create_buffer(wk);

        match wk.dtype {
            DataType::Q4_K => {
                gpu_ctx.mul_mm_q4_k_encode(
                    encoder,
                    &wk_buffer,
                    &normed_f16_buffer,
                    &k_buffer,
                    k_out_size,
                    batch,
                    hidden_size,
                );
            }
            DataType::Q5_K => {
                gpu_ctx.mul_mm_q5_k_encode(
                    encoder,
                    &wk_buffer,
                    &normed_f16_buffer,
                    &k_buffer,
                    k_out_size,
                    batch,
                    hidden_size,
                );
            }
            DataType::Q6_K => {
                gpu_ctx.mul_mm_q6_k_encode(
                    encoder,
                    &wk_buffer,
                    &normed_f16_buffer,
                    &k_buffer,
                    k_out_size,
                    batch,
                    hidden_size,
                );
            }
            _ => panic!("Unsupported dtype {:?}", wk.dtype),
        }

        // Step 5: V projection
        println!("5. V projection (dtype={:?})", wv.dtype);
        let v_buffer = forward_pool.get(batch * v_out_size * 4);
        let wv_buffer = gpu_ctx.get_or_create_buffer(wv);

        match wv.dtype {
            DataType::Q4_K => {
                gpu_ctx.mul_mm_q4_k_encode(
                    encoder,
                    &wv_buffer,
                    &normed_f16_buffer,
                    &v_buffer,
                    v_out_size,
                    batch,
                    hidden_size,
                );
            }
            DataType::Q5_K => {
                gpu_ctx.mul_mm_q5_k_encode(
                    encoder,
                    &wv_buffer,
                    &normed_f16_buffer,
                    &v_buffer,
                    v_out_size,
                    batch,
                    hidden_size,
                );
            }
            DataType::Q6_K => {
                gpu_ctx.mul_mm_q6_k_encode(
                    encoder,
                    &wv_buffer,
                    &normed_f16_buffer,
                    &v_buffer,
                    v_out_size,
                    batch,
                    hidden_size,
                );
            }
            _ => panic!("Unsupported dtype {:?}", wv.dtype),
        }

        // End encoding and commit
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Check results
        println!("\n=== Checking results ===");

        let q_data: Vec<f32> = unsafe {
            let ptr = q_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; batch * q_out_size];
            std::ptr::copy_nonoverlapping(
                ptr,
                data.as_mut_ptr(),
                batch * q_out_size,
            );
            data
        };
        let q_nan = q_data.iter().filter(|x| x.is_nan()).count();
        println!("Q: {} NaN out of {}", q_nan, q_data.len());

        let k_data: Vec<f32> = unsafe {
            let ptr = k_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; batch * k_out_size];
            std::ptr::copy_nonoverlapping(
                ptr,
                data.as_mut_ptr(),
                batch * k_out_size,
            );
            data
        };
        let k_nan = k_data.iter().filter(|x| x.is_nan()).count();
        println!("K: {} NaN out of {}", k_nan, k_data.len());

        let v_data: Vec<f32> = unsafe {
            let ptr = v_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; batch * v_out_size];
            std::ptr::copy_nonoverlapping(
                ptr,
                data.as_mut_ptr(),
                batch * v_out_size,
            );
            data
        };
        let v_nan = v_data.iter().filter(|x| x.is_nan()).count();
        println!("V: {} NaN out of {}", v_nan, v_data.len());
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
