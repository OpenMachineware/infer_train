// Detailed decode timing analysis

use infer_train::cpu::CpuWorkBuffers;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::LlamaModel;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== Decode Timing Analysis ===\n");

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let hidden_size = ctx.config.hidden_size;
    let intermediate_size = ctx.config.intermediate_size;
    let n_layers = ctx.model.layers.len();

    println!(
        "hidden_size={}, intermediate_size={}, n_layers={}",
        hidden_size, intermediate_size, n_layers
    );

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        intermediate_size,
        ctx.config.n_heads,
        ctx.config.n_kv_heads,
        ctx.config.head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        ctx.config.vocab_size,
        0,
        2048,
        n_layers,
    );
    buffers.init_norm_weights(&ctx.model);

    // Warmup
    for _ in 0..3 {
        let _ = forward_detailed(&ctx.model, &[1], &mut kv_cache, &mut buffers);
    }
    kv_cache.clear();

    // Detailed timing
    println!("\n=== Detailed Timing (1 token) ===");
    let _ = forward_detailed(&ctx.model, &[1], &mut kv_cache, &mut buffers);
}

#[cfg(target_arch = "aarch64")]
fn forward_detailed(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut infer_train::inference::KVCache,
    buffers: &mut CpuWorkBuffers,
) -> Option<Vec<f32>> {
    let hidden_size = model.config.hidden_size;
    let n_heads = model.config.n_heads;
    let head_dim = model.config.head_dim;
    let n_kv_heads = model.config.n_kv_heads;

    for &token in tokens {
        // Embedding
        let t_embed = Instant::now();
        model.embedding_lookup_into(token, &mut buffers.hidden)?;
        println!(
            "Embedding: {:.3} ms",
            t_embed.elapsed().as_secs_f64() * 1000.0
        );

        for (layer_idx, layer) in model.layers.iter().enumerate() {
            println!("\n--- Layer {} ---", layer_idx);

            // Attention norm
            let t = Instant::now();
            model.apply_rms_norm_into(
                &buffers.hidden,
                layer.attention_norm.as_ref()?,
                &mut buffers.hidden_alt,
            )?;
            println!(
                "  Attn norm: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );

            // QKV projections
            let t = Instant::now();
            let t1 = Instant::now();
            model.gemv_into(
                &buffers.hidden_alt,
                layer.wq.as_ref()?,
                &mut buffers.q,
            );
            println!(
                "    WQ gemv: {:.3} ms",
                t1.elapsed().as_secs_f64() * 1000.0
            );

            let t1 = Instant::now();
            model.gemv_into(
                &buffers.hidden_alt,
                layer.wk.as_ref()?,
                &mut buffers.k,
            );
            println!(
                "    WK gemv: {:.3} ms",
                t1.elapsed().as_secs_f64() * 1000.0
            );

            let t1 = Instant::now();
            model.gemv_into(
                &buffers.hidden_alt,
                layer.wv.as_ref()?,
                &mut buffers.v,
            );
            println!(
                "    WV gemv: {:.3} ms",
                t1.elapsed().as_secs_f64() * 1000.0
            );
            println!(
                "  QKV total: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );

            // RoPE
            let t = Instant::now();
            model.apply_rope_in_place(
                &mut buffers.q,
                kv_cache.seq_len(),
                n_heads,
                head_dim,
            )?;
            model.apply_rope_in_place(
                &mut buffers.k,
                kv_cache.seq_len(),
                n_kv_heads,
                head_dim,
            )?;
            println!("  RoPE: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

            // KV cache store
            let t = Instant::now();
            kv_cache.set_k(layer_idx, &buffers.k);
            kv_cache.set_v(layer_idx, &buffers.v);
            println!(
                "  KV store: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );

            // Attention
            let t = Instant::now();
            model.flash_attention_decode_into(
                &buffers.q,
                kv_cache,
                layer_idx,
                n_heads,
                head_dim,
                n_kv_heads,
                &mut buffers.hidden_alt,
            )?;
            println!(
                "  Attention: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );

            // Output projection
            let t = Instant::now();
            model.gemv_into(
                &buffers.hidden_alt,
                layer.wo.as_ref()?,
                &mut buffers.hidden,
            );
            println!("  WO gemv: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

            // Residual
            let t = Instant::now();
            for i in 0..hidden_size {
                buffers.hidden[i] += buffers.hidden_saved[i];
            }
            println!(
                "  Residual: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );

            // Save for residual
            buffers.hidden_saved.copy_from_slice(&buffers.hidden);

            // FFN norm
            let t = Instant::now();
            model.apply_rms_norm_into(
                &buffers.hidden,
                layer.ffn_norm.as_ref()?,
                &mut buffers.hidden_alt,
            )?;
            println!(
                "  FFN norm: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );

            // FFN
            let t = Instant::now();
            model.compute_ffn_zero_alloc_single_thread(
                layer,
                &buffers.hidden_alt,
                &mut buffers.ffn_gate,
                &mut buffers.ffn_up,
                &mut buffers.hidden,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            )?;
            println!("  FFN: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

            // Final residual
            let t = Instant::now();
            for i in 0..hidden_size {
                buffers.hidden[i] += buffers.hidden_saved[i];
            }
            println!(
                "  Residual: {:.3} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );
        }

        kv_cache.advance();
    }

    // Final norm and output
    let t = Instant::now();
    model.apply_rms_norm_into(
        &buffers.hidden,
        model.norm.as_ref()?,
        &mut buffers.hidden_alt,
    )?;
    println!("\nFinal norm: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

    let t = Instant::now();
    let vocab_size = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab_size];
    model.gemv_into(&buffers.hidden_alt, model.output.as_ref()?, &mut logits);
    println!("Output gemv: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0);

    Some(logits)
}

#[cfg(not(target_arch = "aarch64"))]
fn forward_detailed(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut infer_train::inference::KVCache,
    _buffers: &mut CpuWorkBuffers,
) -> Option<Vec<f32>> {
    model.forward(tokens, kv_cache)
}
