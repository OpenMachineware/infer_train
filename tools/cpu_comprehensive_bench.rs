// Comprehensive CPU benchmark: prefill + decode
// Usage: cpu-comprehensive-bench <model.gguf> [prefill_tokens] [decode_tokens]

use infer_train::cpu::CpuWorkBuffers;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::LlamaModel;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model.gguf> [prefill_tokens] [decode_tokens]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prefill_tokens: usize =
        args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let decode_tokens: usize =
        args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let hidden_size = ctx.config.hidden_size;
    let intermediate_size = ctx.config.intermediate_size;
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;
    let vocab_size = ctx.config.vocab_size;
    let n_layers = ctx.model.layers.len();

    let mut kv_cache = KVCache::new(&ctx.config, 4096, KVCacheFormat::F32);

    // Initialize buffers
    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        vocab_size,
        0,
        4096,
        n_layers,
    );
    buffers.init_norm_weights(&ctx.model);

    // Warmup
    let warmup_tokens: Vec<u32> = (1..=10).collect();
    let _ =
        forward_batch(&ctx.model, &warmup_tokens, &mut kv_cache, &mut buffers);
    kv_cache.clear();

    // Prefill benchmark
    let prefill_input: Vec<u32> = (1..=prefill_tokens as u32).collect();
    kv_cache.clear();

    let prefill_start = Instant::now();
    let _ =
        forward_batch(&ctx.model, &prefill_input, &mut kv_cache, &mut buffers);
    let prefill_time = prefill_start.elapsed();

    let prefill_tps = prefill_tokens as f64 / prefill_time.as_secs_f64();

    // Decode benchmark
    let decode_start = Instant::now();
    for _ in 0..decode_tokens {
        let token = 100 + (kv_cache.seq_len() as u32) % 100;
        let _ =
            forward_batch(&ctx.model, &[token], &mut kv_cache, &mut buffers);
    }
    let decode_time = decode_start.elapsed();

    let decode_tps = decode_tokens as f64 / decode_time.as_secs_f64();

    // Output results
    println!("model: {}", model_path.split('/').last().unwrap_or(model_path));
    println!("layers: {}", n_layers);
    println!("prefill_tokens: {}", prefill_tokens);
    println!("prefill_time_ms: {:.2}", prefill_time.as_secs_f64() * 1000.0);
    println!("prefill_t/s: {:.2}", prefill_tps);
    println!("decode_tokens: {}", decode_tokens);
    println!("decode_time_ms: {:.2}", decode_time.as_secs_f64() * 1000.0);
    println!("decode_t/s: {:.2}", decode_tps);
}

fn forward_batch(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut infer_train::inference::KVCache,
    buffers: &mut CpuWorkBuffers,
) -> Option<Vec<f32>> {
    let hidden_size = model.config.hidden_size;

    for &token in tokens {
        model.embedding_lookup_into(token, &mut buffers.hidden)?;

        for (layer_idx, layer) in model.layers.iter().enumerate() {
            model.forward_layer_zero_alloc(
                layer, layer_idx, kv_cache, buffers,
            )?;
        }

        kv_cache.advance();
    }

    model.apply_rms_norm_into(
        &buffers.hidden,
        model.norm.as_ref()?,
        &mut buffers.hidden_alt,
    )?;

    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    model.gemv_into(&buffers.hidden_alt, model.output.as_ref()?, &mut logits);

    Some(logits)
}
