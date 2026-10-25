// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test GPU layer 0 only to find NaN source

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

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

        // Test with single token
        let tokens = vec![1u32];

        println!("\n=== Testing embedding lookup ===");
        let hidden =
            ctx.model.embedding_lookup_impl(1).expect("Embedding failed");
        let nan_count = hidden.iter().filter(|x| x.is_nan()).count();
        println!("Embedding: {} values, {} NaN", hidden.len(), nan_count);

        // Upload to GPU
        let hidden_size = ctx.config.hidden_size;
        let hidden_buffer =
            std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
        println!("Hidden buffer uploaded: {} bytes", hidden_buffer.length());

        println!("\n=== Testing Layer 0 ===");
        let layer = &ctx.model.layers[0];

        // 1. RMS Norm
        println!("\n1. RMS Norm");
        let attention_norm =
            layer.attention_norm.as_ref().expect("No attention norm");
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
        let normed_buffer = gpu_ctx
            .rms_norm_f32_gpu(
                &hidden_buffer,
                &norm_weight_buffer,
                1,
                hidden_size,
                ctx.config.rms_norm_eps,
            )
            .expect("RMS norm failed");

        // Download and check
        let normed_data: Vec<f32> = unsafe {
            let ptr = normed_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; hidden_size];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), hidden_size);
            data
        };
        let nan_count = normed_data.iter().filter(|x| x.is_nan()).count();
        let valid: Vec<f32> =
            normed_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  Normed: {} NaN, range [{:.6}, {:.6}]",
            nan_count,
            valid.iter().cloned().fold(f32::INFINITY, f32::min),
            valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // 2. QKV projections
        println!("\n2. QKV projections");
        let wq = layer.wq.as_ref().expect("No wq");
        let wk = layer.wk.as_ref().expect("No wk");
        let wv = layer.wv.as_ref().expect("No wv");

        let q_buffer = ctx
            .model
            .gemv_gpu_buffer_with_fallback(
                &normed_buffer,
                wq,
                wq.shape[1],
                &gpu_ctx,
            )
            .expect("Q projection failed");
        let k_buffer = ctx
            .model
            .gemv_gpu_buffer_with_fallback(
                &normed_buffer,
                wk,
                wk.shape[1],
                &gpu_ctx,
            )
            .expect("K projection failed");
        let v_buffer = ctx
            .model
            .gemv_gpu_buffer_with_fallback(
                &normed_buffer,
                wv,
                wv.shape[1],
                &gpu_ctx,
            )
            .expect("V projection failed");

        // Check Q
        let q_data: Vec<f32> = unsafe {
            let ptr = q_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; wq.shape[1]];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), wq.shape[1]);
            data
        };
        let q_nan = q_data.iter().filter(|x| x.is_nan()).count();
        let q_valid: Vec<f32> =
            q_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  Q: {} NaN, {} valid, range [{:.6}, {:.6}]",
            q_nan,
            q_valid.len(),
            if q_valid.is_empty() {
                f32::NAN
            } else {
                q_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if q_valid.is_empty() {
                f32::NAN
            } else {
                q_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        // Check K
        let k_data: Vec<f32> = unsafe {
            let ptr = k_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; wk.shape[1]];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), wk.shape[1]);
            data
        };
        let k_nan = k_data.iter().filter(|x| x.is_nan()).count();
        let k_valid: Vec<f32> =
            k_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  K: {} NaN, {} valid, range [{:.6}, {:.6}]",
            k_nan,
            k_valid.len(),
            if k_valid.is_empty() {
                f32::NAN
            } else {
                k_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if k_valid.is_empty() {
                f32::NAN
            } else {
                k_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        // Check V
        let v_data: Vec<f32> = unsafe {
            let ptr = v_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; wv.shape[1]];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), wv.shape[1]);
            data
        };
        let v_nan = v_data.iter().filter(|x| x.is_nan()).count();
        let v_valid: Vec<f32> =
            v_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  V: {} NaN, {} valid, range [{:.6}, {:.6}]",
            v_nan,
            v_valid.len(),
            if v_valid.is_empty() {
                f32::NAN
            } else {
                v_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if v_valid.is_empty() {
                f32::NAN
            } else {
                v_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        // 3. RoPE
        println!("\n3. RoPE");
        let head_dim = ctx.config.head_dim;
        let n_heads = wq.shape[1] / head_dim;
        let n_kv_heads = wk.shape[1] / head_dim;

        let q_rope_buffer = gpu_ctx
            .rope_neox_f32_gpu(
                &q_buffer,
                &[0i32],
                1,
                n_heads,
                head_dim,
                ctx.config.rope_freq_base,
                1.0,
            )
            .expect("Q RoPE failed");
        let k_rope_buffer = gpu_ctx
            .rope_neox_f32_gpu(
                &k_buffer,
                &[0i32],
                1,
                n_kv_heads,
                head_dim,
                ctx.config.rope_freq_base,
                1.0,
            )
            .expect("K RoPE failed");

        // Check Q after RoPE
        let q_rope_data: Vec<f32> = unsafe {
            let ptr = q_rope_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; wq.shape[1]];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), wq.shape[1]);
            data
        };
        let q_rope_nan = q_rope_data.iter().filter(|x| x.is_nan()).count();
        let q_rope_valid: Vec<f32> =
            q_rope_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  Q after RoPE: {} NaN, {} valid, range [{:.6}, {:.6}]",
            q_rope_nan,
            q_rope_valid.len(),
            if q_rope_valid.is_empty() {
                f32::NAN
            } else {
                q_rope_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if q_rope_valid.is_empty() {
                f32::NAN
            } else {
                q_rope_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        // Check K after RoPE
        let k_rope_data: Vec<f32> = unsafe {
            let ptr = k_rope_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; wk.shape[1]];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), wk.shape[1]);
            data
        };
        let k_rope_nan = k_rope_data.iter().filter(|x| x.is_nan()).count();
        let k_rope_valid: Vec<f32> =
            k_rope_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  K after RoPE: {} NaN, {} valid, range [{:.6}, {:.6}]",
            k_rope_nan,
            k_rope_valid.len(),
            if k_rope_valid.is_empty() {
                f32::NAN
            } else {
                k_rope_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if k_rope_valid.is_empty() {
                f32::NAN
            } else {
                k_rope_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        // 4. Update KV cache
        println!("\n4. KV Cache Update");
        kv_cache.set_k_buffer(0, &k_rope_buffer);
        kv_cache.set_v_buffer(0, &v_buffer);
        println!("  KV cache updated for layer 0");

        // 5. Flash Attention
        println!("\n5. Flash Attention");
        let k_buffer_from_cache = kv_cache.k_buffer(0);
        let v_buffer_from_cache = kv_cache.v_buffer(0);
        let kv_stride = kv_cache.max_seq();

        let attn_out_buffer = gpu_ctx
            .flash_attn_f16_gpu(
                1, // n_seqs
                n_heads,
                n_kv_heads,
                head_dim,
                1, // seq_len (only 1 token)
                kv_stride,
                &q_rope_buffer,
                &k_buffer_from_cache,
                &v_buffer_from_cache,
            )
            .expect("Flash attention failed");

        // Check attention output
        let attn_data: Vec<f32> = unsafe {
            let ptr = attn_out_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; n_heads * head_dim];
            std::ptr::copy_nonoverlapping(
                ptr,
                data.as_mut_ptr(),
                n_heads * head_dim,
            );
            data
        };
        let attn_nan = attn_data.iter().filter(|x| x.is_nan()).count();
        let attn_valid: Vec<f32> =
            attn_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  Attention output: {} NaN, {} valid, range [{:.6}, {:.6}]",
            attn_nan,
            attn_valid.len(),
            if attn_valid.is_empty() {
                f32::NAN
            } else {
                attn_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if attn_valid.is_empty() {
                f32::NAN
            } else {
                attn_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        // 6. Output projection
        println!("\n6. Output projection (wo)");
        let wo = layer.wo.as_ref().expect("No wo");
        let wo_out_buffer = ctx
            .model
            .gemv_gpu_buffer_with_fallback(
                &attn_out_buffer,
                wo,
                wo.shape[1],
                &gpu_ctx,
            )
            .expect("WO projection failed");

        let wo_data: Vec<f32> = unsafe {
            let ptr = wo_out_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; wo.shape[1]];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), wo.shape[1]);
            data
        };
        let wo_nan = wo_data.iter().filter(|x| x.is_nan()).count();
        let wo_valid: Vec<f32> =
            wo_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  WO output: {} NaN, {} valid, range [{:.6}, {:.6}]",
            wo_nan,
            wo_valid.len(),
            if wo_valid.is_empty() {
                f32::NAN
            } else {
                wo_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if wo_valid.is_empty() {
                f32::NAN
            } else {
                wo_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        // 7. Residual add
        println!("\n7. Residual add");
        let residual_buffer = gpu_ctx
            .add_f32_gpu(&hidden_buffer, &wo_out_buffer, hidden_size)
            .expect("Residual add failed");

        let residual_data: Vec<f32> = unsafe {
            let ptr = residual_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; hidden_size];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), hidden_size);
            data
        };
        let residual_nan = residual_data.iter().filter(|x| x.is_nan()).count();
        let residual_valid: Vec<f32> =
            residual_data.iter().filter(|x| x.is_finite()).cloned().collect();
        println!(
            "  Residual: {} NaN, {} valid, range [{:.6}, {:.6}]",
            residual_nan,
            residual_valid.len(),
            if residual_valid.is_empty() {
                f32::NAN
            } else {
                residual_valid.iter().cloned().fold(f32::INFINITY, f32::min)
            },
            if residual_valid.is_empty() {
                f32::NAN
            } else {
                residual_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
        );

        println!("\n=== Layer 0 test complete ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
