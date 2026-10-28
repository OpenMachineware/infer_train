// SPDX-License-Identifier: AGPL-3.0-or-later
// Trace where zeros come from in DeepSeek standard forward

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    println!("Model: {}", model_path);
    println!(
        "Hidden: {}, Layers: {}, Heads: {}, KV heads: {}",
        ctx.config.hidden_size,
        ctx.config.n_layers,
        ctx.config.n_heads,
        ctx.config.n_kv_heads
    );

    let test_token = 1u32;

    // Test embedding
    let embed = ctx.model.embedding_lookup_impl(test_token).unwrap();
    println!(
        "\nEmbedding (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        embed[0], embed[1], embed[2], embed[3], embed[4]
    );

    // Test layer 0 forward
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let layer = &ctx.model.layers[0];

    // Norm
    let normed = ctx
        .model
        .apply_rms_norm_impl(
            &embed,
            layer.attention_norm.as_ref().unwrap(),
            ctx.config.hidden_size,
        )
        .unwrap();
    println!(
        "After norm (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        normed[0], normed[1], normed[2], normed[3], normed[4]
    );

    // QKV
    let (q, k, v, q_size, k_size, v_size) = ctx
        .model
        .get_qkv_proj(
            layer,
            &normed,
            ctx.config.n_heads,
            ctx.config.n_kv_heads,
            ctx.config.head_dim,
        )
        .unwrap();
    println!(
        "Q (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        q[0], q[1], q[2], q[3], q[4]
    );
    println!(
        "K (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        k[0], k[1], k[2], k[3], k[4]
    );
    println!(
        "V (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        v[0], v[1], v[2], v[3], v[4]
    );

    // Check weight dtypes
    println!("\nWeight dtypes:");
    if let Some(ref wq) = layer.wq {
        println!("  wq: {:?}", wq.dtype);
    }
    if let Some(ref wk) = layer.wk {
        println!("  wk: {:?}", wk.dtype);
    }
    if let Some(ref wv) = layer.wv {
        println!("  wv: {:?}", wv.dtype);
    }
    if let Some(ref wo) = layer.wo {
        println!("  wo: {:?}", wo.dtype);
    }

    // Check if QKV are all zeros
    let q_zeros = q.iter().all(|&x| x == 0.0);
    let k_zeros = k.iter().all(|&x| x == 0.0);
    let v_zeros = v.iter().all(|&x| x == 0.0);
    println!("\nQ all zeros: {}", q_zeros);
    println!("K all zeros: {}", k_zeros);
    println!("V all zeros: {}", v_zeros);

    // Full forward
    let mut kv_full = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits = ctx.forward(&[test_token], &mut kv_full).unwrap();
    println!(
        "\nFull forward logits (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        logits[0], logits[1], logits[2], logits[3], logits[4]
    );
    let logits_zeros = logits.iter().all(|&x| x == 0.0);
    println!("Logits all zeros: {}", logits_zeros);
}
