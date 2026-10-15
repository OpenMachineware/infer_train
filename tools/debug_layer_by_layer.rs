// Debug layer-by-layer comparison between CPU and GPU
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, ModelConfig, llama::LlamaModel};

fn stats(name: &str, data: &[f32]) {
    let num_nan = data.iter().filter(|&&v| v.is_nan()).count();
    let num_inf = data.iter().filter(|&&v| v.is_infinite()).count();
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    println!(
        "  {}: len={}, nan={}, inf={}, range=[{:.4}, {:.4}], mean={:.4}",
        name,
        data.len(),
        num_nan,
        num_inf,
        min,
        max,
        mean
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    // Load model (this will fix config from weight shapes)
    let store = WeightStore::from_gguf(model_path).unwrap();
    let gguf = GgufFile::open(model_path).unwrap();
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);

    // Use corrected config from model
    let config = model.config.clone();

    println!("Layers: {}", model.layers.len());
    println!("Hidden size: {}", config.hidden_size);
    println!(
        "n_heads: {}, n_kv_heads: {}, head_dim: {}",
        config.n_heads, config.n_kv_heads, config.head_dim
    );

    // Test tokens
    let tokens: Vec<u32> = vec![1];
    let hidden_size = config.hidden_size;

    // Print weight dtypes
    if let Some(layer) = model.layers.first() {
        println!("\nLayer 0 weights:");
        if let Some(wq) = layer.wq.as_ref() {
            println!("  wq: {:?}, shape={:?}", wq.dtype, wq.shape);
        }
        if let Some(wk) = layer.wk.as_ref() {
            println!("  wk: {:?}, shape={:?}", wk.dtype, wk.shape);
        }
        if let Some(wv) = layer.wv.as_ref() {
            println!("  wv: {:?}, shape={:?}", wv.dtype, wv.shape);
        }
    }

    // CPU forward
    println!("\n=== CPU Forward ===");
    let mut cpu_kv = KVCache::new(&config, 16, KVCacheFormat::F32);
    let hidden = model.embedding_lookup_impl(tokens[0]).unwrap();
    stats("embedding", &hidden);

    let mut cpu_hidden = hidden.clone();
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        // For layer 1, print MoE-specific info
        if layer_idx == 1 {
            println!("\n=== Layer 1 MoE Debug ===");
            println!("Has MoE: {}", layer.ffn_gate_exps.is_some());
            if let Some(gate_inp) = &layer.ffn_gate_inp {
                println!("gate_inp shape: {:?}", gate_inp.shape);
                // Print first few routing scores
                let weight: &[f32] = bytemuck::cast_slice(gate_inp.bytes());
                println!(
                    "gate_inp first 10 weights per expert 0: {:?}",
                    &weight[0..10]
                );
            }
            if let Some(gate_exps) = &layer.ffn_gate_exps {
                println!(
                    "gate_exps shape: {:?}, dtype: {:?}",
                    gate_exps.shape, gate_exps.dtype
                );
                println!("gate_exps bytes len: {}", gate_exps.bytes().len());
            }
            if let Some(down_exps) = &layer.ffn_down_exps {
                println!(
                    "down_exps shape: {:?}, dtype: {:?}",
                    down_exps.shape, down_exps.dtype
                );
            }
            println!("n_experts_used: {}", config.n_experts_used);
        }

        cpu_hidden = model
            .forward_layer(
                layer,
                layer_idx,
                &cpu_hidden,
                &mut cpu_kv,
                config.n_heads,
                config.head_dim,
                config.n_kv_heads,
            )
            .expect(&format!("Layer {} forward failed", layer_idx));

        // Print all layers
        stats(&format!("layer {} out", layer_idx), &cpu_hidden);
    }

    // GPU forward (if available)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("\n=== GPU Forward ===");
        let gpu_ctx =
            infer_train::quant::vec_dot::gpu::metal::MetalContext::new()
                .unwrap();
        let mut gpu_kv = infer_train::inference::KVCacheGPU::new(
            gpu_ctx.device(),
            &config,
            16,
        );

        let hidden = model.embedding_lookup_impl(tokens[0]).unwrap();
        let hidden_buffer =
            std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

        // Debug: check initial hidden buffer
        println!("\nDebug: Initial hidden buffer");
        let hidden_check: Vec<f32> = unsafe {
            let ptr = hidden_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; 10];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), 10);
            data
        };
        println!("  First 10 values: {:?}", &hidden_check);

        // Test RMS norm on its own
        println!("\nDebug: Testing RMS norm");
        let layer = &model.layers[0];
        let attn_norm = layer.attention_norm.as_ref().unwrap();
        let norm_buffer = gpu_ctx.get_or_create_buffer(attn_norm);
        let normed = gpu_ctx
            .rms_norm_f32_gpu(
                &hidden_buffer,
                &norm_buffer,
                1,
                hidden_size,
                1e-5,
            )
            .unwrap();

        let normed_check: Vec<f32> = unsafe {
            let ptr = normed.contents() as *const f32;
            let mut data = vec![0.0f32; 10];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), 10);
            data
        };
        println!("  RMS norm output first 10: {:?}", &normed_check);

        // Test GEMV on its own
        println!("\nDebug: Testing GEMV");
        let wq = layer.wq.as_ref().unwrap();
        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        println!("  wq dtype: {:?}, shape: {:?}", wq.dtype, wq.shape);
        let q_out_size = wq.shape[1];

        // Check if the dtype is supported
        let supported = matches!(
            wq.dtype,
            infer_train::model::tensor::DataType::Q4_0
                | infer_train::model::tensor::DataType::Q4_1
                | infer_train::model::tensor::DataType::Q5_0
                | infer_train::model::tensor::DataType::Q5_1
                | infer_train::model::tensor::DataType::Q8_0
                | infer_train::model::tensor::DataType::Q2_K
                | infer_train::model::tensor::DataType::Q3_K
                | infer_train::model::tensor::DataType::Q4_K
                | infer_train::model::tensor::DataType::Q5_K
                | infer_train::model::tensor::DataType::Q6_K
        );
        println!("  wq dtype supported in GPU GEMV: {}", supported);

        let q_buffer = model
            .gemv_gpu_buffer(
                &normed, &wq_buffer, q_out_size, wq.dtype, &gpu_ctx,
            )
            .unwrap();

        let q_check: Vec<f32> = unsafe {
            let ptr = q_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; 10];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), 10);
            data
        };
        println!("  Q output first 10: {:?}", &q_check);

        println!("\nContinuing with full layer processing...");
        let mut gpu_hidden_buffer = hidden_buffer;
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            gpu_hidden_buffer = model
                .forward_layer_gpu_buffer(
                    layer,
                    layer_idx,
                    &gpu_hidden_buffer,
                    &mut gpu_kv,
                    config.n_heads,
                    config.head_dim,
                    config.n_kv_heads,
                    &gpu_ctx,
                )
                .unwrap();

            // Download and compare
            if layer_idx < 5 || layer_idx >= model.layers.len() - 3 {
                let gpu_data: Vec<f32> = unsafe {
                    let ptr = gpu_hidden_buffer.contents() as *const f32;
                    let mut data = vec![0.0f32; hidden_size];
                    std::ptr::copy_nonoverlapping(
                        ptr,
                        data.as_mut_ptr(),
                        hidden_size,
                    );
                    data
                };
                stats(&format!("layer {} out", layer_idx), &gpu_data);
            } else if layer_idx == 5 {
                println!("  ... (skipping middle layers)");
            }
        }

        gpu_kv.advance();
    }
}
