// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Debug prefill NaN layer by layer

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::model::gguf::GgufFile;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("Model: {}", model_path);

        // Parse GGUF to check parameters
        let gguf = GgufFile::open(model_path).expect("Failed to parse GGUF");
        let arch = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| {
                if let infer_train::model::gguf::MetadataValue::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let arch_prefix = format!("{}.", arch);
        let rope_freq_base: f32 = gguf
            .metadata
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(&arch_prefix).map(|s| (s.to_string(), v.clone()))
            })
            .find(|(k, _)| k == "rope.freq_base")
            .and_then(|(_, v)| {
                if let infer_train::model::gguf::MetadataValue::Float32(f) = v {
                    Some(f)
                } else {
                    None
                }
            })
            .unwrap_or(10000.0);

        println!("Architecture: {}", arch);
        println!("RoPE freq_base: {}", rope_freq_base);

        // Load model for GPU
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load GPU model");

        println!("\nConfig from model:");
        println!("  rope_freq_base: {}", ctx.config.rope_freq_base);
        println!("  rms_norm_eps: {}", ctx.config.rms_norm_eps);
        println!("  hidden_size: {}", ctx.config.hidden_size);
        println!("  n_heads: {}", ctx.config.n_heads);
        println!("  n_kv_heads: {}", ctx.config.n_kv_heads);
        println!("  head_dim: {}", ctx.config.head_dim);

        // Check layer 0 weights
        if let Some(layer) = ctx.model.layers.first() {
            println!("\nLayer 0 weight dtypes:");
            if let Some(ref wq) = layer.wq {
                println!("  wq: {:?}", wq.dtype);
            }
            if let Some(ref wk) = layer.wk {
                println!("  wk: {:?}", wk.dtype);
            }
            if let Some(ref wv) = layer.wv {
                println!("  wv: {:?}", wv.dtype);
            }
        }

        // Test CPU first
        println!("\n=== CPU inference (baseline) ===");
        let cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
            .expect("Failed to load CPU model");
        let mut cpu_kv = infer_train::inference::KVCache::new(
            &cpu_ctx.config,
            2048,
            infer_train::inference::KVCacheFormat::F32,
        );
        let cpu_tokens = vec![1u32, 2u32];
        let cpu_logits = cpu_ctx
            .model
            .forward(&cpu_tokens, &mut cpu_kv)
            .expect("CPU forward failed");
        let cpu_nan = cpu_logits.iter().filter(|x| x.is_nan()).count();
        println!("CPU logits: {} NaN out of {}", cpu_nan, cpu_logits.len());

        // Test GPU with prefill
        println!("\n=== GPU prefill with 2 tokens ===");
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let mut kv_cache = KVCacheGPU::new(gpu_ctx.device(), &ctx.config, 2048);

        let tokens = vec![1u32, 2u32];
        let batch = tokens.len();

        // Manual embedding lookup
        let hidden_size = ctx.config.hidden_size;
        let mut all_embeddings = vec![0.0f32; batch * hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            let embedding = ctx
                .model
                .embedding_lookup_impl(token)
                .expect("Embedding failed");
            all_embeddings[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&embedding);
        }

        // Check embeddings for NaN
        let embed_nan = all_embeddings.iter().filter(|x| x.is_nan()).count();
        println!(
            "Embeddings: {} NaN out of {}",
            embed_nan,
            all_embeddings.len()
        );

        // Upload to GPU
        let embed_buffer =
            std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
                all_embeddings.as_ptr() as *const std::ffi::c_void,
                (batch * hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

        // Create command buffer and encoder for layer 0 only
        let queue = gpu_ctx.queue();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();

        // Setup forward pool
        use std::cell::RefCell;
        use std::rc::Rc;
        let reuse_pool: Rc<RefCell<Vec<Rc<metal::Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool = infer_train::buffer::ForwardBufferPool::new(
            gpu_ctx.device().clone(),
            reuse_pool.clone(),
        );

        // Run just layer 0
        let layer = &ctx.model.layers[0];
        let layer_input = forward_pool.register(embed_buffer.clone());

        let result = ctx.model.forward_layer_prefill_batched(
            layer,
            0,
            &layer_input,
            batch,
            &mut kv_cache,
            &gpu_ctx,
            &mut forward_pool,
            encoder,
        );

        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        match result {
            Some(output_buffer) => {
                // Download and check
                let output_data: Vec<f32> = unsafe {
                    let ptr = output_buffer.contents() as *const f32;
                    let mut data = vec![0.0f32; batch * hidden_size];
                    std::ptr::copy_nonoverlapping(
                        ptr,
                        data.as_mut_ptr(),
                        batch * hidden_size,
                    );
                    data
                };

                let nan_count =
                    output_data.iter().filter(|x| x.is_nan()).count();
                let valid: Vec<f32> = output_data
                    .iter()
                    .filter(|x| x.is_finite())
                    .cloned()
                    .collect();
                println!(
                    "Layer 0 output: {} NaN out of {}",
                    nan_count,
                    output_data.len()
                );
                if !valid.is_empty() {
                    println!(
                        "Valid range: [{:.6}, {:.6}]",
                        valid.iter().cloned().fold(f32::INFINITY, f32::min),
                        valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    );
                }
            }
            None => {
                println!("Layer 0 forward failed!");
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
