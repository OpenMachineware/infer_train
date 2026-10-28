// SPDX-License-Identifier: AGPL-3.0-or-later
// Compare first layer outputs between standard and graph

use infer_train::inference::{Device, InferenceContext, KVCache, KVCacheFormat};
use infer_train::model::LlamaModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let mut ctx = InferenceContext::from_gguf(&args[1], Device::Cpu)
        .expect("Failed to load model");

    let test_token = 1u32;

    // Trace standard forward through layer 0
    println!("=== Standard forward through layer 0 ===");
    let mut kv_cache_std = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    let hidden = ctx.model.embedding_lookup_impl(test_token).expect("Embedding failed");
    println!("Embedding sum: {:.4}", hidden.iter().sum::<f32>());

    let layer0 = &ctx.model.layers[0];
    let normed = ctx.model.apply_rms_norm_impl(&hidden, layer0.attention_norm.as_ref().unwrap(), hidden.len()).unwrap();
    println!("After norm sum: {:.4}", normed.iter().sum::<f32>());

    // Get QKV
    let (q, k, v, _, _, _) = ctx.model.get_qkv_proj(layer0, &normed, ctx.config.n_heads, ctx.config.n_kv_heads, ctx.config.head_dim).unwrap();
    println!("Q sum: {:.4}, len: {}", q.iter().sum::<f32>(), q.len());
    println!("K sum: {:.4}, len: {}", k.iter().sum::<f32>(), k.len());
    println!("V sum: {:.4}, len: {}", v.iter().sum::<f32>(), v.len());

    // Apply RoPE
    let q_rope = ctx.model.apply_rope(&q, 0, ctx.config.n_heads, ctx.config.head_dim).unwrap();
    let k_rope = ctx.model.apply_rope(&k, 0, ctx.config.n_kv_heads, ctx.config.head_dim).unwrap();
    println!("Q after RoPE sum: {:.4}", q_rope.iter().sum::<f32>());
    println!("K after RoPE sum: {:.4}", k_rope.iter().sum::<f32>());

    // Attention
    let attn_out = ctx.model.attention(&q_rope, &k_rope, &v, 0, &mut kv_cache_std, ctx.config.n_heads, ctx.config.head_dim, ctx.config.n_kv_heads).unwrap();
    println!("Attention output sum: {:.4}", attn_out.iter().sum::<f32>());

    // Output projection
    let attn_proj = ctx.model.gemv_impl(&attn_out, layer0.wo.as_ref().unwrap(), ctx.config.hidden_size).unwrap();
    println!("Output projection sum: {:.4}", attn_proj.iter().sum::<f32>());

    // Residual
    let residual: Vec<f32> = hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();
    println!("Residual sum: {:.4}", residual.iter().sum::<f32>());

    // Now trace graph execution
    println!("\n=== Graph execution ===");
    ctx.model.build_optimized_dag();
    let dag = ctx.model.optimized_dag.as_ref().unwrap();

    // Find the attention output tensor ID
    for (i, node_id) in dag.execution_order.iter().take(15).enumerate() {
        let node = &dag.dag.nodes[*node_id];
        println!("[{}] {:?}", i, node.op);
    }
}
