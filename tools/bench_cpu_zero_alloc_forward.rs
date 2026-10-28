// Benchmark zero-allocation CPU forward pass

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
    println!("=== CPU Zero-Allocation Forward Benchmark ===\n");
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
        "hidden_size: {}, intermediate_size: {}, layers: {}",
        hidden_size, intermediate_size, n_layers
    );
    println!(
        "n_heads: {}, n_kv_heads: {}, head_dim: {}",
        n_heads, n_kv_heads, head_dim
    );

    // Check fused QKV dimensions
    let qkv_out_size = if let Some(ref qkv) = ctx.model.layers[0].attn_qkv {
        println!("Fused QKV shape: {:?}", qkv.shape);
        let q_size = n_heads * head_dim;
        let k_size = n_kv_heads * head_dim;
        let v_size = n_kv_heads * head_dim;
        println!(
            "Expected QKV output: {} (Q:{} + K:{} + V:{})",
            q_size + k_size + v_size,
            q_size,
            k_size,
            v_size
        );
        // Use actual tensor dimension
        qkv.shape[1]
    } else {
        0 // Will use calculated size
    };

    // Create pre-allocated buffers
    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        vocab_size,
        qkv_out_size,
        2048, // max_seq_len
        n_layers,
    );

    println!(
        "Buffer sizes: q8k_input={} blocks, x_padded={} floats",
        buffers.q8k_input.len() / 264,
        buffers.x_padded.len()
    );
    println!(
        "q_buffer={}, k_buffer={}, v_buffer={}, attn_out={}",
        buffers.q_buffer.len(),
        buffers.k_buffer.len(),
        buffers.v_buffer.len(),
        buffers.attn_out.len()
    );

    // Initialize pre-dequantized norm weights
    buffers.init_norm_weights(&ctx.model);
    println!(
        "Norm weights initialized: {} elements",
        buffers.norm_weights.len()
    );

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Warmup with regular forward
    println!("\n=== Warmup ===");
    let tokens: Vec<u32> = vec![1, 2, 3];
    let _ = ctx.model.forward(&tokens, &mut kv_cache);
    kv_cache.clear();
    println!("Warmup complete");

    // Benchmark regular forward
    println!("\n=== Regular Forward (20 tokens) ===");
    let mut decode_tokens = vec![1u32];

    let start = std::time::Instant::now();
    for i in 0..20 {
        match ctx.model.forward(&decode_tokens, &mut kv_cache) {
            Some(_) => {}
            None => {
                eprintln!("Forward failed at token {}", i);
                break;
            }
        }
        decode_tokens.clear();
        decode_tokens.push(100 + (kv_cache.seq_len() as u32) % 100);
    }
    let elapsed_regular = start.elapsed();
    let regular_tps = 20.0 / elapsed_regular.as_secs_f64();

    println!("Time: {:?}", elapsed_regular);
    println!("Speed: {:.2} t/s", regular_tps);

    // Benchmark zero-allocation forward
    println!("\n=== Zero-Allocation Forward (20 tokens) ===");
    kv_cache.clear();
    decode_tokens = vec![1u32];

    let start = std::time::Instant::now();
    for i in 0..20 {
        // Use zero-allocation forward
        match forward_zero_alloc(
            &ctx.model,
            &decode_tokens,
            &mut kv_cache,
            &mut buffers,
        ) {
            Some(_) => {}
            None => {
                eprintln!("Zero-alloc forward failed at token {}", i);
                break;
            }
        }
        decode_tokens.clear();
        decode_tokens.push(100 + (kv_cache.seq_len() as u32) % 100);
    }
    let elapsed_zero_alloc = start.elapsed();
    let zero_alloc_tps = 20.0 / elapsed_zero_alloc.as_secs_f64();

    println!("Time: {:?}", elapsed_zero_alloc);
    println!("Speed: {:.2} t/s", zero_alloc_tps);

    // Comparison
    println!("\n=== Comparison ===");
    println!("Regular: {:.2} t/s", regular_tps);
    println!("Zero-alloc: {:.2} t/s", zero_alloc_tps);
    if zero_alloc_tps > regular_tps {
        println!("Improvement: {:.2}x", zero_alloc_tps / regular_tps);
    } else {
        println!("Regression: {:.2}x", regular_tps / zero_alloc_tps);
    }
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
