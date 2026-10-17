// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug full GPU forward pass - find where NaN appears

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
    use std::rc::Rc;

    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

    println!("Loading model on CPU...");
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let model = ctx.model();
    let config = ctx.config.clone();

    println!("Model loaded: {} layers", model.layers.len());

    // Create Metal context and KV cache
    let gpu_ctx = infer_train::quant::vec_dot::gpu::metal::MetalContext::new()
        .expect("Failed to create MetalContext");
    let mut kv_cache = KVCacheGPU::new(gpu_ctx.device(), &config, 2048);

    let tokens: Vec<u32> = vec![1, 2, 3];

    // Process each token
    let mut last_hidden_buffer: Option<Rc<metal::Buffer>> = None;

    for (token_idx, &token) in tokens.iter().enumerate() {
        println!("\n=== Processing Token {} ===", token_idx);

        // Embedding on CPU
        let hidden =
            model.embedding_lookup_impl(token).expect("Embedding failed");
        println!(
            "Embedding: {} values, range=[{}, {}]",
            hidden.len(),
            hidden.iter().cloned().fold(f32::INFINITY, f32::min),
            hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Upload to GPU
        let mut hidden_buffer = Rc::new(gpu_ctx.device().new_buffer_with_data(
            hidden.as_ptr() as *const std::ffi::c_void,
            (hidden.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Process all layers
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let n_heads = config.n_heads;
            let head_dim = config.head_dim;
            let n_kv_heads = config.n_kv_heads;

            hidden_buffer = model
                .forward_layer_gpu_buffer(
                    layer,
                    layer_idx,
                    &hidden_buffer,
                    &mut kv_cache,
                    n_heads,
                    head_dim,
                    n_kv_heads,
                    &gpu_ctx,
                )
                .expect(&format!("Layer {} failed", layer_idx));

            // Check for NaN
            let hidden: Vec<f32> = unsafe {
                let ptr = hidden_buffer.contents() as *const f32;
                let mut data = vec![0.0f32; config.hidden_size];
                std::ptr::copy_nonoverlapping(
                    ptr,
                    data.as_mut_ptr(),
                    config.hidden_size,
                );
                data
            };

            let nan_count = hidden.iter().filter(|&&v| v.is_nan()).count();
            let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            if nan_count > 0 {
                println!(
                    "\n!!! Token {}, Layer {} produced {} NaNs !!!",
                    token_idx, layer_idx, nan_count
                );
                println!("  Range: [{}, {}]", min, max);
                println!("  First 10: {:?}", &hidden[..10]);
                return;
            }

            if token_idx == 0 && layer_idx % 5 == 0 {
                println!("Layer {}: range=[{}, {}]", layer_idx, min, max);
            }
        }

        // Save the last hidden buffer
        last_hidden_buffer = Some(hidden_buffer);

        kv_cache.advance();
    }

    // Use the last hidden buffer
    let hidden_buffer = last_hidden_buffer.expect("No hidden buffer");

    // Final norm
    println!("\n=== Final Norm ===");
    let final_norm = model.norm.as_ref().expect("No final norm");
    let final_norm_buffer = gpu_ctx.get_or_create_buffer(final_norm);

    let final_hidden_buffer = gpu_ctx
        .rms_norm_f32_gpu(
            &hidden_buffer,
            &final_norm_buffer,
            1,
            config.hidden_size,
            1e-5,
        )
        .expect("Final norm failed");

    let final_hidden: Vec<f32> = unsafe {
        let ptr = final_hidden_buffer.contents() as *const f32;
        let mut data = vec![0.0f32; config.hidden_size];
        std::ptr::copy_nonoverlapping(
            ptr,
            data.as_mut_ptr(),
            config.hidden_size,
        );
        data
    };

    let nan_count = final_hidden.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "Final hidden: {} values, {} NaN, range=[{}, {}]",
        final_hidden.len(),
        nan_count,
        final_hidden.iter().cloned().fold(f32::INFINITY, f32::min),
        final_hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Output projection
    println!("\n=== Output Projection ===");
    let output = model.output.as_ref().expect("No output");
    println!("Output: dtype={:?}, shape={:?}", output.dtype, output.shape);

    let is_transposed =
        output.shape.len() == 2 && output.shape[0] == config.hidden_size;
    let actual_vocab_size =
        if is_transposed { output.shape[1] } else { output.shape[0] };
    println!(
        "actual_vocab_size: {}, is_transposed: {}",
        actual_vocab_size, is_transposed
    );

    let logits_buffer = model
        .gemv_gpu_buffer_with_fallback(
            &final_hidden_buffer,
            output,
            actual_vocab_size,
            &gpu_ctx,
        )
        .expect("Output projection failed");

    let logits: Vec<f32> = unsafe {
        let ptr = logits_buffer.contents() as *const f32;
        let mut data = vec![0.0f32; actual_vocab_size];
        std::ptr::copy_nonoverlapping(
            ptr,
            data.as_mut_ptr(),
            actual_vocab_size,
        );
        data
    };

    let nan_count = logits.iter().filter(|&&v| v.is_nan()).count();
    let positive = logits.iter().filter(|&&v| v > 0.0).count();
    let negative = logits.iter().filter(|&&v| v < 0.0).count();

    println!(
        "Logits: {} values, {} NaN, {} positive, {} negative",
        logits.len(),
        nan_count,
        positive,
        negative
    );
    println!(
        "  Range: [{}, {}]",
        logits.iter().cloned().fold(f32::INFINITY, f32::min),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("  First 10: {:?}", &logits[..10]);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("Metal not available on this platform");
}
