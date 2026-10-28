// SPDX-License-Identifier: AGPL-3.0-or-later
// Trace layer-by-layer execution to find divergence point

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
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let test_token = 1u32;

    // Build DAG
    ctx.model.build_optimized_dag();
    let dag = ctx.model.optimized_dag.as_ref().unwrap();

    // Compare embeddings
    println!("=== Embedding ===");
    let embed = ctx.model.embedding_lookup_impl(test_token).unwrap();
    println!("  Embedding[0..5]: {:?}", &embed[..5]);

    // Now manually trace first layer operations
    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;
    let head_dim = ctx.config.head_dim;
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;

    // Step 1: Attention norm
    println!("\n=== Step 1: Attention norm ===");
    let attn_norm = layer.attention_norm.as_ref().unwrap();
    let normed =
        ctx.model.apply_rms_norm_impl(&embed, attn_norm, hidden_size).unwrap();
    println!("  Normed[0..5]: {:?}", &normed[..5]);

    // Step 2: Q/K/V projections
    println!("\n=== Step 2: Q/K/V projections ===");
    let (q, k, v, q_out_size, k_out_size, v_out_size) = ctx
        .model
        .get_qkv_proj(layer, &normed, n_heads, n_kv_heads, head_dim)
        .unwrap();
    println!("  Q size: {}, Q[0..5]: {:?}", q.len(), &q[..5]);
    println!("  K size: {}, K[0..5]: {:?}", k.len(), &k[..5]);
    println!("  V size: {}, V[0..5]: {:?}", v.len(), &v[..5]);

    // Step 3: RoPE
    println!("\n=== Step 3: RoPE ===");
    let pos = 0;
    let q_rope = ctx.model.apply_rope(&q, pos, n_heads, head_dim).unwrap();
    let k_rope = ctx.model.apply_rope(&k, pos, n_kv_heads, head_dim).unwrap();
    println!("  Q_rope[0..5]: {:?}", &q_rope[..5]);
    println!("  K_rope[0..5]: {:?}", &k_rope[..5]);

    // Step 4: Attention (with KV cache)
    println!("\n=== Step 4: Attention ===");
    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let attn_out = ctx
        .model
        .attention(
            &q_rope,
            &k_rope,
            &v,
            0,
            &mut kv_cache,
            n_heads,
            head_dim,
            n_kv_heads,
        )
        .unwrap();
    println!("  Attn_out[0..5]: {:?}", &attn_out[..5]);
    println!("  KV cache seq_len: {}", kv_cache.seq_len());

    // Step 5: Output projection
    println!("\n=== Step 5: Output projection (wo) ===");
    let wo_out = if let Some(wo) = &layer.wo {
        ctx.model.gemv(&attn_out, wo, hidden_size).unwrap()
    } else {
        attn_out.clone()
    };
    println!("  Wo_out[0..5]: {:?}", &wo_out[..5]);

    // Step 6: Residual add
    println!("\n=== Step 6: Residual add ===");
    let residual: Vec<f32> =
        embed.iter().zip(wo_out.iter()).map(|(a, b)| a + b).collect();
    println!("  Residual[0..5]: {:?}", &residual[..5]);

    // Step 7: FFN norm
    println!("\n=== Step 7: FFN norm ===");
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let ffn_normed = ctx
        .model
        .apply_rms_norm_impl(&residual, ffn_norm, hidden_size)
        .unwrap();
    println!("  FFN_normed[0..5]: {:?}", &ffn_normed[..5]);

    // Step 8: Gate projection
    println!("\n=== Step 8: Gate/Up projections ===");
    let gate = if let Some(w1) = &layer.w1 {
        ctx.model.gemv(&ffn_normed, w1, w1.shape[1]).unwrap()
    } else {
        vec![0.0f32; 0]
    };
    let up = if let Some(w3) = &layer.w3 {
        ctx.model.gemv(&ffn_normed, w3, w3.shape[1]).unwrap()
    } else {
        vec![0.0f32; 0]
    };
    println!("  Gate[0..5]: {:?}", &gate[..5.min(gate.len())]);
    println!("  Up[0..5]: {:?}", &up[..5.min(up.len())]);

    // Step 9: SiLU and mul
    println!("\n=== Step 9: SiLU and mul ===");
    // Correct: SiLU is applied to UP, not gate!
    // hidden = gate * silu(up)
    let ffn_hidden: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| g * (u / (1.0 + (-u).exp())))
        .collect();
    println!(
        "  FFN_hidden[0..5]: {:?}",
        &ffn_hidden[..5.min(ffn_hidden.len())]
    );

    // Step 10: Down projection
    println!("\n=== Step 10: Down projection ===");
    let ffn_out = if let Some(w2) = &layer.w2 {
        ctx.model.gemv(&ffn_hidden, w2, hidden_size).unwrap()
    } else {
        vec![0.0f32; hidden_size]
    };
    println!("  FFN_out[0..5]: {:?}", &ffn_out[..5]);

    // Step 11: Final residual add
    println!("\n=== Step 11: Final residual add ===");
    let layer_out: Vec<f32> =
        residual.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
    println!("  Layer_out[0..5]: {:?}", &layer_out[..5]);

    // Compare with standard forward's first layer output
    println!("\n=== Comparison with standard forward ===");
    let mut kv_std = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let std_layer_out = ctx
        .model
        .forward_layer(
            layer,
            0,
            &embed,
            &mut kv_std,
            n_heads,
            head_dim,
            n_kv_heads,
        )
        .unwrap();
    println!("  Std_layer_out[0..5]: {:?}", &std_layer_out[..5]);

    let mut max_diff = 0.0f32;
    for i in 0..layer_out.len().min(std_layer_out.len()) {
        let diff = (layer_out[i] - std_layer_out[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("  Max diff: {}", max_diff);
}
