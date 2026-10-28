// Debug the profile tool timing discrepancy

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
    let mut kv_cache = KVCache::new(&ctx.config, 100, KVCacheFormat::F32);

    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;

    let hidden = ctx.model.embedding_lookup_impl(1).expect("Embedding failed");

    // Test 1: Measure just the FFN call (like profile-cpu-detailed does)
    println!("=== Test 1: Exact same code as profile-cpu-detailed ===");
    let normed = ctx
        .model
        .apply_rms_norm_impl(
            &hidden,
            layer.attention_norm.as_ref().unwrap(),
            hidden_size,
        )
        .unwrap();

    let (q, k, v, _, _, _) = ctx
        .model
        .get_qkv_proj(
            layer,
            &normed,
            ctx.config.n_heads,
            ctx.config.n_kv_heads,
            ctx.config.head_dim,
        )
        .unwrap();

    let q_rope = ctx
        .model
        .apply_rope(&q, 0, ctx.config.n_heads, ctx.config.head_dim)
        .unwrap();
    let k_rope = ctx
        .model
        .apply_rope(&k, 0, ctx.config.n_kv_heads, ctx.config.head_dim)
        .unwrap();

    let attn_out = ctx
        .model
        .attention(
            &q_rope,
            &k_rope,
            &v,
            0,
            &mut kv_cache,
            ctx.config.n_heads,
            ctx.config.head_dim,
            ctx.config.n_kv_heads,
        )
        .unwrap();

    let attn_proj = ctx
        .model
        .gemv_impl(&attn_out, layer.wo.as_ref().unwrap(), hidden_size)
        .unwrap();
    let residual: Vec<f32> =
        hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();
    let ffn_normed = ctx
        .model
        .apply_rms_norm_impl(
            &residual,
            layer.ffn_norm.as_ref().unwrap(),
            hidden_size,
        )
        .unwrap();

    // Now measure the FFN - EXACTLY like profile-cpu-detailed does
    let t = std::time::Instant::now();
    let _ffn_out =
        ctx.model.compute_ffn(layer, &ffn_normed, hidden_size).unwrap();
    let ffn_time = t.elapsed();
    println!("FFN time (single call, cold): {:?}", ffn_time);

    // Warm up and measure again
    for _ in 0..10 {
        let _ = ctx.model.compute_ffn(layer, &ffn_normed, hidden_size).unwrap();
    }
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &ffn_normed, hidden_size).unwrap();
    }
    println!("FFN time (avg of 20, warm): {:?}", t.elapsed() / 20);

    // Test 2: Measure with fresh context (like profile-cpu-detailed creates)
    println!("\n=== Test 2: Fresh context ===");
    let mut ctx2 = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    let layer2 = &ctx2.model.layers[0];
    let ffn_normed2: Vec<f32> = vec![0.5; hidden_size];

    let t = std::time::Instant::now();
    let _ = ctx2.model.compute_ffn(layer2, &ffn_normed2, hidden_size).unwrap();
    println!("FFN time (fresh context, cold): {:?}", t.elapsed());

    // Test 3: Check if there's something special about the residual value
    println!("\n=== Test 3: Different input values ===");
    let zeros: Vec<f32> = vec![0.0; hidden_size];
    let ones: Vec<f32> = vec![1.0; hidden_size];
    let random: Vec<f32> =
        (0..hidden_size).map(|i| (i as f32 % 10.0) - 5.0).collect();

    for _ in 0..10 {
        let _ = ctx.model.compute_ffn(layer, &zeros, hidden_size);
    }
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &zeros, hidden_size);
    }
    println!("FFN with zeros: {:?}", t.elapsed() / 20);

    for _ in 0..10 {
        let _ = ctx.model.compute_ffn(layer, &ones, hidden_size);
    }
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &ones, hidden_size);
    }
    println!("FFN with ones: {:?}", t.elapsed() / 20);

    for _ in 0..10 {
        let _ = ctx.model.compute_ffn(layer, &random, hidden_size);
    }
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &random, hidden_size);
    }
    println!("FFN with random: {:?}", t.elapsed() / 20);

    // Test 4: Run profile-cpu-detailed's exact code
    println!("\n=== Test 4: Run full profile like profile-cpu-detailed ===");

    let mut ctx3 = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    let mut kv_cache3 = KVCache::new(&ctx3.config, 100, KVCacheFormat::F32);
    let hidden3 =
        ctx3.model.embedding_lookup_impl(1).expect("Embedding failed");
    let layer3 = &ctx3.model.layers[0];

    // Reproduce the exact steps from profile-cpu-detailed.rs
    let t1 = std::time::Instant::now();
    let normed3 = ctx3
        .model
        .apply_rms_norm_impl(
            &hidden3,
            layer3.attention_norm.as_ref().unwrap(),
            hidden_size,
        )
        .unwrap();
    let attn_norm_time = t1.elapsed();

    let t2 = std::time::Instant::now();
    let (q3, k3, v3, _, _, _) = ctx3
        .model
        .get_qkv_proj(
            layer3,
            &normed3,
            ctx3.config.n_heads,
            ctx3.config.n_kv_heads,
            ctx3.config.head_dim,
        )
        .unwrap();
    let qkv_time = t2.elapsed();

    let t3 = std::time::Instant::now();
    let q_rope3 = ctx3
        .model
        .apply_rope(&q3, 0, ctx3.config.n_heads, ctx3.config.head_dim)
        .unwrap();
    let k_rope3 = ctx3
        .model
        .apply_rope(&k3, 0, ctx3.config.n_kv_heads, ctx3.config.head_dim)
        .unwrap();
    let rope_time = t3.elapsed();

    let t4 = std::time::Instant::now();
    let attn_out3 = ctx3
        .model
        .attention(
            &q_rope3,
            &k_rope3,
            &v3,
            0,
            &mut kv_cache3,
            ctx3.config.n_heads,
            ctx3.config.head_dim,
            ctx3.config.n_kv_heads,
        )
        .unwrap();
    let attn_time = t4.elapsed();

    let t5 = std::time::Instant::now();
    let attn_proj3 = ctx3
        .model
        .gemv_impl(&attn_out3, layer3.wo.as_ref().unwrap(), hidden_size)
        .unwrap();
    let wo_time = t5.elapsed();

    let t6 = std::time::Instant::now();
    let residual3: Vec<f32> =
        hidden3.iter().zip(attn_proj3.iter()).map(|(h, a)| h + a).collect();
    let residual1_time = t6.elapsed();

    let t7 = std::time::Instant::now();
    let ffn_normed3 = ctx3
        .model
        .apply_rms_norm_impl(
            &residual3,
            layer3.ffn_norm.as_ref().unwrap(),
            hidden_size,
        )
        .unwrap();
    let ffn_norm_time = t7.elapsed();

    let t8 = std::time::Instant::now();
    let _ffn_out3 =
        ctx3.model.compute_ffn(layer3, &ffn_normed3, hidden_size).unwrap();
    let ffn_time = t8.elapsed();

    println!("Attn norm: {:?}", attn_norm_time);
    println!("QKV: {:?}", qkv_time);
    println!("RoPE: {:?}", rope_time);
    println!("Attention: {:?}", attn_time);
    println!("Output proj: {:?}", wo_time);
    println!("Residual: {:?}", residual1_time);
    println!("FFN norm: {:?}", ffn_norm_time);
    println!("FFN: {:?}", ffn_time);
    println!(
        "Total: {:?}",
        attn_norm_time
            + qkv_time
            + rope_time
            + attn_time
            + wo_time
            + residual1_time
            + ffn_norm_time
            + ffn_time
    );
}
