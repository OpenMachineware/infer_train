// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug GPU forward pass to find where it fails

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::inference::{Device, InferenceContext};
    use std::rc::Rc;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    println!("✓ Model loaded");
    println!("  Layers: {}", ctx.layer_count());
    println!(
        "  Config: vocab={}, hidden={}",
        ctx.config.vocab_size, ctx.config.hidden_size
    );
    println!("  n_heads: {}", ctx.config.n_heads);
    println!("  n_kv_heads: {}", ctx.config.n_kv_heads);
    println!("  head_dim: {}", ctx.config.head_dim);

    let tokens: Vec<u32> = vec![1, 2, 3];
    println!("\nInput tokens: {:?}", tokens);

    // Manual forward pass with debug output
    let model = &ctx.model;
    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let kv_cache = ctx.gpu_kv_cache.as_mut().expect("No KV cache");

    let hidden_size = model.config.hidden_size;
    let n_heads = model.config.n_heads;
    let head_dim = model.config.head_dim;
    let n_kv_heads = model.config.n_kv_heads;

    println!("\n=== Processing tokens ===");

    let mut last_hidden_buffer = None;

    for (token_idx, &token) in tokens.iter().enumerate() {
        println!("\n--- Token {} (id={}) ---", token_idx, token);

        // 1. Embedding lookup
        println!("1. Embedding lookup...");
        let hidden =
            model.embedding_lookup_impl(token).expect("Embedding failed");
        println!(
            "   Hidden: len={}, min={:.4}, max={:.4}",
            hidden.len(),
            hidden.iter().cloned().fold(f32::INFINITY, f32::min),
            hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        // Upload to GPU
        println!("2. Uploading to GPU...");
        let hidden_buffer = Rc::new(gpu_ctx.device().new_buffer_with_data(
            hidden.as_ptr() as *const std::ffi::c_void,
            (hidden_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        println!("   Buffer size: {} bytes", hidden_buffer.length());

        // Process layers
        let mut hidden_buffer = hidden_buffer;
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            println!("\n3. Layer {}...", layer_idx);

            // Check layer components
            println!(
                "   attn_norm: {}",
                if layer.attention_norm.is_some() { "✓" } else { "✗" }
            );
            println!("   wq: {}", if layer.wq.is_some() { "✓" } else { "✗" });
            println!("   wk: {}", if layer.wk.is_some() { "✓" } else { "✗" });
            println!("   wv: {}", if layer.wv.is_some() { "✓" } else { "✗" });
            println!("   wo: {}", if layer.wo.is_some() { "✓" } else { "✗" });
            println!(
                "   ffn_norm: {}",
                if layer.ffn_norm.is_some() { "✓" } else { "✗" }
            );
            println!(
                "   w1 (gate): {}",
                if layer.w1.is_some() { "✓" } else { "✗" }
            );
            println!(
                "   w3 (up): {}",
                if layer.w3.is_some() { "✓" } else { "✗" }
            );
            println!(
                "   w2 (down): {}",
                if layer.w2.is_some() { "✓" } else { "✗" }
            );

            // Try forward_layer_gpu_buffer
            hidden_buffer = match model.forward_layer_gpu_buffer(
                layer,
                layer_idx,
                &hidden_buffer,
                kv_cache,
                n_heads,
                head_dim,
                n_kv_heads,
                gpu_ctx,
            ) {
                Some(buffer) => {
                    println!("   ✓ Layer {} succeeded", layer_idx);
                    buffer
                }
                None => {
                    println!("   ✗ Layer {} FAILED", layer_idx);
                    return;
                }
            };
        }

        // Advance KV cache
        kv_cache.advance();
        println!("\n4. KV cache advanced to position {}", kv_cache.seq_len());

        // Save for final projection
        last_hidden_buffer = Some(hidden_buffer);
    }

    let hidden_buffer = last_hidden_buffer.expect("No hidden buffer");

    // Final norm
    println!("\n=== Final projection ===");
    let final_norm = model.norm.as_ref().expect("No final norm");
    println!("Final norm: {:?}", final_norm.shape);

    let final_norm_buffer = gpu_ctx.get_or_create_buffer(final_norm);
    let final_hidden_buffer = gpu_ctx
        .rms_norm_f32_gpu(
            &hidden_buffer,
            &final_norm_buffer,
            1,
            hidden_size,
            1e-5,
        )
        .expect("Final norm failed");
    println!("✓ Final norm succeeded");

    // Output projection
    let output = model.output.as_ref().expect("No output layer");
    println!(
        "\nOutput layer: dtype={:?}, shape={:?}",
        output.dtype, output.shape
    );

    let vocab_size = if output.shape[0] == hidden_size {
        output.shape[1]
    } else {
        output.shape[0]
    };
    println!("Vocab size: {}", vocab_size);

    let logits = model
        .gemv_gpu_buffer_with_fallback(
            &final_hidden_buffer,
            output,
            vocab_size,
            gpu_ctx,
        )
        .expect("Output projection failed");
    println!("✓ Output projection succeeded");

    // Download logits
    let mut logits_data = vec![0.0f32; vocab_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            logits.contents() as *const f32,
            logits_data.as_mut_ptr(),
            vocab_size,
        );
    }

    println!("\n=== Results ===");
    println!("Logits: {} values", logits_data.len());
    println!("Non-zero: {}", logits_data.iter().filter(|&&v| v != 0.0).count());
    println!("NaN: {}", logits_data.iter().filter(|&&v| v.is_nan()).count());
    println!(
        "Range: [{:.4}, {:.4}]",
        logits_data.iter().cloned().fold(f32::INFINITY, f32::min),
        logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("First 10: {:?}", &logits_data[..10]);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support");
}
