// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug GPU inference step by step

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::inference::{
        Device, InferenceContext, KVCache, KVCacheFormat,
    };
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use std::rc::Rc;

    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

    println!("Loading model on CPU...");
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let model = ctx.model();
    let config = ctx.config.clone();

    println!(
        "Model loaded: {} layers, {}M params",
        model.layers.len(),
        ctx.param_count() / 1_000_000
    );
    println!(
        "Config: hidden={}, vocab={}, heads={}, kv_heads={}",
        config.hidden_size,
        config.vocab_size,
        config.n_heads,
        config.n_kv_heads
    );

    // Create Metal context
    let gpu_ctx = MetalContext::new().expect("Failed to create MetalContext");

    // Test step by step
    let token: u32 = 1;

    // Step 1: Embedding
    println!("\n=== Step 1: Embedding ===");
    let hidden = model.embedding_lookup_impl(token).expect("Embedding failed");
    let nan_count = hidden.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "Embedding: {} values, {} NaN, range=[{}, {}]",
        hidden.len(),
        nan_count,
        hidden.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Upload to GPU
    let hidden_buffer = Rc::new(gpu_ctx.device().new_buffer_with_data(
        hidden.as_ptr() as *const std::ffi::c_void,
        (hidden.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Step 2: First layer RMS norm
    println!("\n=== Step 2: Layer 0 RMS Norm ===");
    let layer0 = &model.layers[0];
    let attn_norm = layer0.attention_norm.as_ref().expect("No attn norm");

    let norm_buffer = gpu_ctx.get_or_create_buffer(attn_norm);

    let normed_buffer = gpu_ctx
        .rms_norm_f32_gpu(
            &hidden_buffer,
            &norm_buffer,
            1,
            config.hidden_size,
            1e-5,
        )
        .expect("RMS norm failed");

    // Read back
    let normed: Vec<f32> = unsafe {
        let ptr = normed_buffer.contents() as *const f32;
        let mut data = vec![0.0f32; config.hidden_size];
        std::ptr::copy_nonoverlapping(
            ptr,
            data.as_mut_ptr(),
            config.hidden_size,
        );
        data
    };

    let nan_count = normed.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "RMS Norm: {} values, {} NaN, range=[{}, {}]",
        normed.len(),
        nan_count,
        normed.iter().cloned().fold(f32::INFINITY, f32::min),
        normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("  First 10: {:?}", &normed[..10]);

    // Step 3: Q projection (GEMV)
    println!("\n=== Step 3: Q Projection ===");
    let wq = layer0.wq.as_ref().expect("No wq");
    println!("  wq: dtype={:?}, shape={:?}", wq.dtype, wq.shape);

    let q_out_size = wq.shape[1];
    let wq_buffer = gpu_ctx.get_or_create_buffer(wq);

    let q_buffer = model
        .gemv_gpu_buffer(
            &normed_buffer,
            &wq_buffer,
            q_out_size,
            wq.dtype,
            &gpu_ctx,
        )
        .expect("Q projection failed");

    // Read back
    let q: Vec<f32> = unsafe {
        let ptr = q_buffer.contents() as *const f32;
        let mut data = vec![0.0f32; q_out_size];
        std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), q_out_size);
        data
    };

    let nan_count = q.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "Q: {} values, {} NaN, range=[{}, {}]",
        q.len(),
        nan_count,
        q.iter().cloned().fold(f32::INFINITY, f32::min),
        q.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("  First 10: {:?}", &q[..10.min(q.len())]);

    // Step 4: K projection
    println!("\n=== Step 4: K Projection ===");
    let wk = layer0.wk.as_ref().expect("No wk");
    println!("  wk: dtype={:?}, shape={:?}", wk.dtype, wk.shape);

    let k_out_size = wk.shape[1];
    let wk_buffer = gpu_ctx.get_or_create_buffer(wk);

    let k_buffer = model
        .gemv_gpu_buffer(
            &normed_buffer,
            &wk_buffer,
            k_out_size,
            wk.dtype,
            &gpu_ctx,
        )
        .expect("K projection failed");

    let k: Vec<f32> = unsafe {
        let ptr = k_buffer.contents() as *const f32;
        let mut data = vec![0.0f32; k_out_size];
        std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), k_out_size);
        data
    };

    let nan_count = k.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "K: {} values, {} NaN, range=[{}, {}]",
        k.len(),
        nan_count,
        k.iter().cloned().fold(f32::INFINITY, f32::min),
        k.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Step 5: V projection
    println!("\n=== Step 5: V Projection ===");
    let wv = layer0.wv.as_ref().expect("No wv");
    println!("  wv: dtype={:?}, shape={:?}", wv.dtype, wv.shape);

    let v_out_size = wv.shape[1];
    let wv_buffer = gpu_ctx.get_or_create_buffer(wv);

    let v_buffer = model
        .gemv_gpu_buffer(
            &normed_buffer,
            &wv_buffer,
            v_out_size,
            wv.dtype,
            &gpu_ctx,
        )
        .expect("V projection failed");

    let v: Vec<f32> = unsafe {
        let ptr = v_buffer.contents() as *const f32;
        let mut data = vec![0.0f32; v_out_size];
        std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), v_out_size);
        data
    };

    let nan_count = v.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "V: {} values, {} NaN, range=[{}, {}]",
        v.len(),
        nan_count,
        v.iter().cloned().fold(f32::INFINITY, f32::min),
        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Step 6: Compare with CPU
    println!("\n=== Step 6: Compare with CPU ===");
    let mut kv_cache = KVCache::new(&config, 16, KVCacheFormat::F32);

    // CPU computation
    let cpu_hidden =
        model.embedding_lookup_impl(token).expect("Embedding failed");
    let cpu_norm = model
        .apply_rms_norm(&cpu_hidden, attn_norm, config.hidden_size)
        .expect("RMS norm failed");

    let cpu_q = model.gemv(&cpu_norm, wq, q_out_size).expect("CPU Q failed");
    let cpu_k = model.gemv(&cpu_norm, wk, k_out_size).expect("CPU K failed");
    let cpu_v = model.gemv(&cpu_norm, wv, v_out_size).expect("CPU V failed");

    println!(
        "\nCPU Q: range=[{}, {}]",
        cpu_q.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "CPU K: range=[{}, {}]",
        cpu_k.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_k.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "CPU V: range=[{}, {}]",
        cpu_v.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Compare
    let q_diff: f32 =
        q.iter().zip(cpu_q.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / q.len() as f32;
    println!("\nQ mean diff: {}", q_diff);

    let k_diff: f32 =
        k.iter().zip(cpu_k.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / k.len() as f32;
    println!("K mean diff: {}", k_diff);

    let v_diff: f32 =
        v.iter().zip(cpu_v.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / v.len() as f32;
    println!("V mean diff: {}", v_diff);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("Metal not available on this platform");
}
