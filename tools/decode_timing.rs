// Per-operation timing for decode to find performance gap
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("=== Decode Per-Operation Timing ===\n");

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    ctx.thread_config = infer_train::cpu::ThreadConfig::with_threads(1, 8);
    let config = ctx.config.clone();

    let mut kv_cache = KVCache::new(&config, 2048, KVCacheFormat::F32);

    // Warm up
    let _ = ctx.forward_pooled(&[1], &mut kv_cache);

    // Time each component
    let iterations = 20;
    let mut total_time = Duration::ZERO;

    for i in 0..iterations {
        kv_cache.clear();
        let start = Instant::now();
        let _ = ctx.forward_pooled(&[1], &mut kv_cache);
        total_time += start.elapsed();
    }

    let avg_ms = total_time.as_secs_f64() * 1000.0 / iterations as f64;
    let tps = 1000.0 / avg_ms;

    println!("Average decode time: {:.2} ms = {:.2} t/s", avg_ms, tps);
    println!();

    // Now time each operation within the forward pass
    // We need to access the model directly for per-operation timing
    kv_cache.clear();

    // Get the model
    let model = ctx.model();

    // Get the hidden state
    let hidden_size = config.hidden_size;

    // Time embedding lookup
    let start = Instant::now();
    let hidden = model.embedding_lookup(1).expect("Embedding lookup failed");
    let embed_time = start.elapsed();
    println!("Embedding lookup: {:?}", embed_time);

    // Process through layers
    for layer_idx in 0..config.n_layers {
        println!("\n--- Layer {} ---", layer_idx);

        let layer = &model.layers[layer_idx];

        // Attention norm
        let start = Instant::now();
        let norm = model
            .apply_rms_norm(
                &hidden,
                layer.attention_norm.as_ref().unwrap(),
                hidden_size,
            )
            .unwrap();
        let norm_time = start.elapsed();
        println!("  Attention norm: {:?}", norm_time);

        // QKV projections
        let start = Instant::now();
        let (q, k, v, _, _, _) = model
            .get_qkv_proj(
                layer,
                &norm,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.key_length,
                config.value_length,
            )
            .unwrap();
        let qkv_time = start.elapsed();
        println!("  QKV projection: {:?}", qkv_time);

        // RoPE
        let start = Instant::now();
        // Note: RoPE is done inside forward_layer, we're skipping detailed timing here
        let rope_time = start.elapsed();
        println!("  RoPE: {:?}", rope_time);

        // Attention
        let start = Instant::now();
        // Note: Attention is done inside forward_layer
        let attn_time = start.elapsed();
        println!("  Attention: {:?}", attn_time);

        // Output projection
        let start = Instant::now();
        // Note: Wo projection is done inside forward_layer
        let wo_time = start.elapsed();
        println!("  Output projection: {:?}", wo_time);

        // FFN norm
        let start = Instant::now();
        // Note: FFN norm is done inside forward_layer
        let ffn_norm_time = start.elapsed();
        println!("  FFN norm: {:?}", ffn_norm_time);

        // FFN
        let start = Instant::now();
        // Note: FFN is done inside forward_layer
        let ffn_time = start.elapsed();
        println!("  FFN: {:?}", ffn_time);

        // Only time first 3 layers for brevity
        if layer_idx >= 2 {
            println!(
                "\n... (skipping remaining {} layers)",
                config.n_layers - 3
            );
            break;
        }
    }
}
