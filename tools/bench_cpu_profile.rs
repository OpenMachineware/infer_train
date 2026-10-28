// Detailed CPU inference profiling

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Inference Detailed Profiling ===\n");
    println!("Model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Model loaded successfully");

    let config = &ctx.config;
    println!(
        "Config: hidden={}, layers={}, heads={}, vocab={}",
        config.hidden_size, config.n_layers, config.n_heads, config.vocab_size
    );

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Prefill benchmark
    println!("\n=== Prefill Benchmark ===");
    let prefill_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let prefill_start = std::time::Instant::now();
    let _ = ctx.model.forward(&prefill_tokens, &mut kv_cache);
    let prefill_elapsed = prefill_start.elapsed();

    println!(
        "Prefill: {} tokens in {:?} ({:.2} t/s)",
        prefill_tokens.len(),
        prefill_elapsed,
        prefill_tokens.len() as f64 / prefill_elapsed.as_secs_f64()
    );
    println!(
        "Prefill time per token: {:?}",
        prefill_elapsed / prefill_tokens.len() as u32
    );

    // Decode benchmark with detailed timing
    println!("\n=== Decode Benchmark (first 10 tokens) ===");
    let mut decode_tokens = vec![1u32];
    let mut decode_times = Vec::new();

    for i in 0..10 {
        let token_start = std::time::Instant::now();
        match ctx.model.forward(&decode_tokens, &mut kv_cache) {
            Some(_logits) => {
                let token_time = token_start.elapsed();
                decode_times.push(token_time);
                println!(
                    "Token {}: {:?} ({:.2} t/s)",
                    i + 1,
                    token_time,
                    1.0 / token_time.as_secs_f64()
                );
            }
            None => {
                eprintln!("Forward failed at token {}", i);
                break;
            }
        }
        decode_tokens.clear();
        decode_tokens.push(100 + (kv_cache.seq_len() as u32) % 100);
    }

    // Calculate bandwidth
    // For Q5_K: each weight is ~0.5 bytes, activations are f32 (4 bytes)
    // Each token processes:
    // - Q/K/V projections: 3 * hidden * hidden bytes read
    // - FFN: 3 * hidden * intermediate bytes read
    // - Output: vocab * hidden bytes read
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let vocab = config.vocab_size;
    let n_layers = config.n_layers;

    // Q5_K: ~0.625 bytes per weight (5 bits + overhead)
    let bytes_per_weight = 0.625;
    let bytes_per_activation = 4.0; // f32

    // Total bytes per token:
    // - Embedding: 1 lookup (negligible)
    // - RMS Norm per layer: 2 * hidden * 4 bytes (input + output)
    // - Q/K/V: 3 * hidden * hidden * 0.625 weights + hidden * 4 input
    // - Attention: hidden * 4 output
    // - FFN gate/up: 2 * hidden * intermediate * 0.625 + hidden * 4
    // - FFN down: hidden * intermediate * 0.625 + intermediate * 4
    // - Output projection: vocab * hidden * 0.625 + hidden * 4

    let weight_bytes_per_token = n_layers as f64
        * (
            // Q/K/V weights
            3.0 * hidden as f64 * hidden as f64 * bytes_per_weight
            // FFN weights
            + 3.0 * hidden as f64 * intermediate as f64 * bytes_per_weight
        )
        + vocab as f64 * hidden as f64 * bytes_per_weight; // Output projection

    // Activation bytes (input vectors)
    let activation_bytes_per_token = n_layers as f64
        * (
            // Input to Q/K/V
            hidden as f64 * bytes_per_activation
            // Input to FFN
            + hidden as f64 * bytes_per_activation
            + intermediate as f64 * bytes_per_activation
        )
        + hidden as f64 * bytes_per_activation; // Output projection input

    let total_bytes_per_token =
        weight_bytes_per_token + activation_bytes_per_token;
    println!("\n=== Bandwidth Analysis ===",);
    println!(
        "Estimated bytes per token: {:.2} MB (weights: {:.2} MB, activations: {:.2} MB)",
        total_bytes_per_token / 1e6,
        weight_bytes_per_token / 1e6,
        activation_bytes_per_token / 1e6
    );

    // Average decode time
    let avg_decode_time = decode_times.iter().sum::<std::time::Duration>()
        / decode_times.len() as u32;
    println!("Average decode time: {:?}", avg_decode_time);

    let bandwidth_gbps =
        total_bytes_per_token / avg_decode_time.as_secs_f64() / 1e9;
    println!("Estimated bandwidth: {:.2} GB/s", bandwidth_gbps);

    // Apple Silicon M4 has ~100 GB/s memory bandwidth
    let theoretical_bandwidth = 100.0; // GB/s
    let bandwidth_utilization = bandwidth_gbps / theoretical_bandwidth * 100.0;
    println!(
        "Bandwidth utilization: {:.1}% (assuming {} GB/s theoretical)",
        bandwidth_utilization, theoretical_bandwidth
    );

    // Compare with target
    println!("\n=== Performance Target ===");
    let target_tps = 97.0; // llama.cpp baseline
    let current_tps = 1.0 / avg_decode_time.as_secs_f64();
    let gap = target_tps / current_tps;
    println!(
        "Current: {:.2} t/s, Target: {:.0} t/s, Gap: {:.1}x",
        current_tps, target_tps, gap
    );

    // If bandwidth is low, it's not the kernel issue
    if bandwidth_utilization < 20.0 {
        println!(
            "\n⚠️  Bandwidth utilization is LOW ({:.1}%)",
            bandwidth_utilization
        );
        println!("    This suggests the issue is NOT the GEMV kernels.");
        println!("    Possible causes:");
        println!("    - Thread synchronization overhead");
        println!("    - Memory allocation during inference");
        println!("    - Non-GEMV operations bottleneck");
    } else {
        println!(
            "\n✓ Bandwidth utilization is reasonable ({:.1}%)",
            bandwidth_utilization
        );
        println!("  Kernel optimization may help further.");
    }
}
