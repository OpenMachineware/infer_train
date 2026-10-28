// Profile forward pass by measuring total time

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== Profile Forward Pass ===\n");

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!(
        "Model: {} layers, hidden={}",
        ctx.model.layers.len(),
        ctx.model.config.hidden_size
    );
    println!(
        "n_heads={}, head_dim={}, n_kv_heads={}",
        ctx.model.config.n_heads,
        ctx.model.config.head_dim,
        ctx.model.config.n_kv_heads
    );

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Warmup
    println!("\nWarmup...");
    let _ = ctx.model.forward(&[1, 2, 3], &mut kv_cache);
    kv_cache.clear();
    println!("Done\n");

    // Profile single token
    println!("=== Single Token Profile ===");

    let n_runs = 20;
    let mut total_times = Vec::new();
    let mut layer_times: Vec<Vec<u128>> =
        (0..ctx.model.layers.len()).map(|_| Vec::new()).collect();

    for run in 0..n_runs {
        kv_cache.clear();

        let t0 = Instant::now();

        // Embedding
        let mut hidden = ctx.model.embedding_lookup_impl(1).unwrap();

        // Process each layer
        for (layer_idx, _layer) in ctx.model.layers.iter().enumerate() {
            let t_layer = Instant::now();
            hidden = ctx
                .model
                .forward_layer(
                    &ctx.model.layers[layer_idx],
                    layer_idx,
                    &hidden,
                    &mut kv_cache,
                    ctx.model.config.n_heads,
                    ctx.model.config.head_dim,
                    ctx.model.config.n_kv_heads,
                )
                .unwrap();
            layer_times[layer_idx].push(t_layer.elapsed().as_nanos());
        }

        kv_cache.advance();

        // Final norm and output
        let _logits = ctx.model.forward(
            &[1],
            &mut KVCache::new(&ctx.config, 2048, KVCacheFormat::F32),
        );

        total_times.push(t0.elapsed().as_nanos());
    }

    // Print results
    println!("\n=== Results ({} runs) ===", n_runs);
    let avg_ns: u128 = total_times.iter().sum::<u128>() / n_runs;
    println!("Total forward: {:.2} ms", avg_ns as f64 / 1_000_000.0);

    println!("\n=== Per-Layer Time ===");
    for (i, times) in layer_times.iter().enumerate() {
        let avg = times.iter().sum::<u128>() / n_runs;
        if i < 5 || i >= ctx.model.layers.len() - 2 {
            println!("Layer {:2}: {:.3} ms", i, avg as f64 / 1_000_000.0);
        } else if i == 5 {
            println!("...");
        }
    }

    // Compare with llama.cpp
    println!("\n=== Comparison ===");
    let our_ms = avg_ns as f64 / 1_000_000.0;
    let llama_cpp_ms = 31.2; // From benchmark
    println!(
        "Our decode:  {:.2} ms/token ({:.2} t/s)",
        our_ms,
        1000.0 / our_ms
    );
    println!(
        "llama.cpp:  {:.2} ms/token ({:.2} t/s)",
        llama_cpp_ms,
        1000.0 / llama_cpp_ms
    );
    println!("Gap: {:.1}x slower", our_ms / llama_cpp_ms);
}
