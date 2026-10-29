// Simple CPU inference benchmark (zero-allocation)

use infer_train::cpu::CpuWorkBuffers;
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
        );
        println!(
            "Warmup token {}: {} ms",
            i,
            token_start.elapsed().as_millis()
        );
    }
    kv_cache.clear();
    println!("Warmup complete: {} ms", warmup_start.elapsed().as_millis());

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
) -> Option<Vec<f32>> {
    let hidden_size = model.config.hidden_size;

    for &token in tokens {
        // 1. Embedding lookup into buffers.hidden
        model.embedding_lookup_into(token, &mut buffers.hidden)?;

        // 2. Process through layers
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            model.forward_layer_zero_alloc(
                layer, layer_idx, kv_cache, buffers,
            )?;
        }

        kv_cache.advance();
    }

    // 3. Final norm
    model.apply_rms_norm_into(
        &buffers.hidden,
        model.norm.as_ref()?,
        &mut buffers.hidden_alt,
    )?;

    // 4. Output projection (still allocates for logits)
    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    model.gemv_into(&buffers.hidden_alt, model.output.as_ref()?, &mut logits);

    Some(logits)
}

#[cfg(not(target_arch = "aarch64"))]
fn forward_zero_alloc(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut infer_train::inference::KVCache,
    _buffers: &mut CpuWorkBuffers,
) -> Option<Vec<f32>> {
    model.forward(tokens, kv_cache)
}
