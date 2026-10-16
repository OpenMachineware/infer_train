// Debug GPU forward layer by layer
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::inference::{Device, InferenceContext, KVCache, KVCacheFormat};
    use metal::MTLResourceOptions;

    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    println!("Model: {}", model_path);

    // Load on GPU
    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");
    println!(
        "Config: hidden={}, heads={}, n_kv_heads={}, head_dim={}",
        gpu_ctx.config.hidden_size,
        gpu_ctx.config.n_heads,
        gpu_ctx.config.n_kv_heads,
        gpu_ctx.config.head_dim
    );

    // Load on CPU for comparison
    let cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load CPU model");
    let mut cpu_kv = KVCache::new(&cpu_ctx.config, 16, KVCacheFormat::F32);

    // Get embedding
    let token: u32 = 1;
    let embed_cpu = cpu_ctx.model.embedding_lookup_impl(token).expect("Embedding failed");
    println!("\nEmbedding: len={}, first 3: {:?}", embed_cpu.len(), &embed_cpu[..3]);

    // Process each layer
    let mut hidden_cpu = embed_cpu.clone();
    for (layer_idx, layer) in cpu_ctx.model.layers.iter().enumerate() {
        println!("\n=== Layer {} ===", layer_idx);

        // CPU forward
        let cpu_start = std::time::Instant::now();
        let layer_out_cpu = cpu_ctx.model.forward_layer(
            layer,
            layer_idx,
            &hidden_cpu,
            &mut cpu_kv,
            cpu_ctx.config.n_heads,
            cpu_ctx.config.head_dim,
            cpu_ctx.config.n_kv_heads,
        ).expect(&format!("CPU layer {} failed", layer_idx));
        let cpu_time = cpu_start.elapsed();
        println!("CPU: {} ms", cpu_time.as_millis());

        // Check output
        let min = layer_out_cpu.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = layer_out_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let nan_count = layer_out_cpu.iter().filter(|&&x| x.is_nan()).count();
        println!("CPU output: min={:.4}, max={:.4}, nan={}", min, max, nan_count);
        println!("CPU first 3: {:?}", &layer_out_cpu[..3]);

        hidden_cpu = layer_out_cpu;

        // Only test first 3 layers to save time
        if layer_idx >= 2 {
            break;
        }
    }

    // Now test GPU forward
    println!("\n=== GPU Full Forward ===");
    let gpu_start = std::time::Instant::now();
    let logits_gpu = gpu_ctx.forward_gpu_batched(&[token]).expect("GPU forward failed");
    let gpu_time = gpu_start.elapsed();
    println!("GPU time: {} ms", gpu_time.as_millis());

    let min = logits_gpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits_gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let zero_count = logits_gpu.iter().filter(|&&x| x == 0.0).count();
    println!("GPU logits: min={:.4}, max={:.4}, zeros={}/{}", min, max, zero_count, logits_gpu.len());
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This tool requires Metal support");
}
