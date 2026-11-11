// Simple CPU inference benchmark (zero-allocation)

use infer_train::cpu::{CpuWorkBuffers, ThreadConfig};
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::LlamaModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Inference Benchmark (Zero-Allocation) ===\n");
    println!("Model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Model loaded successfully");

    let hidden_size = ctx.config.hidden_size;
    let intermediate_size = ctx.config.intermediate_size;
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;
    let vocab_size = ctx.config.vocab_size;
    let n_layers = ctx.model.layers.len();

    println!(
        "Config: hidden={}, intermediate={}, layers={}",
        hidden_size, intermediate_size, n_layers
    );
    println!(
        "Heads: n_heads={}, n_kv_heads={}, head_dim={}",
        n_heads, n_kv_heads, head_dim
    );

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Initialize buffers for zero-allocation
    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        vocab_size,
        0,    // qkv_out_size
        2048, // max_seq_len
        n_layers,
    );

    // Initialize pre-dequantized norm weights
    buffers.init_norm_weights(&ctx.model);

    // Initialize thread configuration for multi-thread decode
    let thread_config = ThreadConfig::default();
    println!(
        "Threads: decode={}, prefill={}",
        thread_config.n_threads_decode, thread_config.n_threads_prefill
    );

    // Warmup
    println!("\n=== Warmup (3 tokens) ===");
    let tokens: Vec<u32> = vec![1, 2, 3];
    let warmup_start = std::time::Instant::now();
    for (i, &token) in tokens.iter().enumerate() {
        let token_start = std::time::Instant::now();
        let _ = forward_zero_alloc(
            &ctx.model,
            &[token],
            &mut kv_cache,
            &mut buffers,
            &thread_config,
        );
        println!(
            "Warmup token {}: {} ms",
            i,
            token_start.elapsed().as_millis()
        );
    }
    kv_cache.clear();
    println!("Warmup complete: {} ms", warmup_start.elapsed().as_millis());

    // Benchmark prefill (128 tokens)
    println!("\n=== Prefill Benchmark (128 tokens) ===");
    let prefill_tokens: Vec<u32> = (0..128).collect();
    let prefill_start = std::time::Instant::now();
    match forward_zero_alloc(
        &ctx.model,
        &prefill_tokens,
        &mut kv_cache,
        &mut buffers,
        &thread_config,
    ) {
        Some(_logits) => {
            let elapsed = prefill_start.elapsed();
            println!("Prefill tokens: 128");
            println!("Prefill time: {:?}", elapsed);
            println!("Prefill speed: {:.2} t/s", 128.0 / elapsed.as_secs_f64());
        }
        None => {
            eprintln!("Prefill failed");
        }
    }

    // Benchmark decode
    println!("\n=== Decode Benchmark (10 tokens) ===");
    let mut decode_tokens = vec![1u32];

    let start = std::time::Instant::now();
    for i in 0..10 {
        let token_start = std::time::Instant::now();
        match forward_zero_alloc(
            &ctx.model,
            &decode_tokens,
            &mut kv_cache,
            &mut buffers,
            &thread_config,
        ) {
            Some(_logits) => {
                let token_time = token_start.elapsed().as_millis();
                println!("Token {}: {} ms", i, token_time);
            }
            None => {
                eprintln!("Forward failed at token {}", i);
                break;
            }
        }
        decode_tokens.clear();
        decode_tokens.push(100 + (kv_cache.seq_len() as u32) % 100);
    }
    let elapsed = start.elapsed();

    println!("Tokens: 10");
    println!("Time: {:?}", elapsed);
    println!("Speed: {:.2} t/s", 10.0 / elapsed.as_secs_f64());
}

/// Zero-allocation forward pass
#[cfg(target_arch = "aarch64")]
fn forward_zero_alloc(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut infer_train::inference::KVCache,
    buffers: &mut CpuWorkBuffers,
    thread_config: &ThreadConfig,
) -> Option<Vec<f32>> {
    // Use forward_pooled_with_threads for multi-thread decode
    model.forward_pooled_with_threads(tokens, kv_cache, buffers, thread_config)
}

#[cfg(not(target_arch = "aarch64"))]
fn forward_zero_alloc(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut infer_train::inference::KVCache,
    _buffers: &mut CpuWorkBuffers,
    _thread_config: &ThreadConfig,
) -> Option<Vec<f32>> {
    model.forward(tokens, kv_cache)
}
