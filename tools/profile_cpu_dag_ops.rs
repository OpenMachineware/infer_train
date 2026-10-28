// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Profile each operation in CPU forward to find bottlenecks

use infer_train::inference::{KVCache, KVCacheFormat};
use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, LlamaModel, ModelConfig};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("Loading model from {}...", model_path);
    let store =
        WeightStore::from_gguf(model_path).expect("Failed to load weights");
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);

    let hidden_size = model.config.hidden_size;
    let n_layers = model.layers.len();

    println!("Model: hidden={}, layers={}", hidden_size, n_layers);

    // Profile single token forward, operation by operation
    let token = 1u32;
    let mut kv_cache = KVCache::new(&model.config, 100, KVCacheFormat::F32);

    println!("\n=== Profiling single token forward ===\n");

    // 1. Embedding
    let t = Instant::now();
    let hidden = model.embedding_lookup_impl(token).expect("Embedding failed");
    println!(
        "[1] Embedding: {:.2} ms, output size: {}",
        t.elapsed().as_secs_f64() * 1000.0,
        hidden.len()
    );

    let mut hidden = hidden;

    // Process each layer
    for layer_idx in 0..n_layers {
        println!("\n--- Layer {} ---", layer_idx);
        let layer = &model.layers[layer_idx];

        // Attention norm
        let t = Instant::now();
        let normed = model
            .apply_rms_norm_impl(
                &hidden,
                layer.attention_norm.as_ref().unwrap(),
                hidden_size,
            )
            .unwrap();
        println!("[A1] RMSNorm: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

        // Q projection
        let q_out_size =
            layer.wq.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);
        let t = Instant::now();
        let q = model
            .gemv_impl(&normed, layer.wq.as_ref().unwrap(), q_out_size)
            .unwrap();
        println!(
            "[A2] Q GEMV ({}->{}): {:.2} ms",
            hidden_size,
            q_out_size,
            t.elapsed().as_secs_f64() * 1000.0
        );

        // K projection
        let k_out_size =
            layer.wk.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);
        let t = Instant::now();
        let k = model
            .gemv_impl(&normed, layer.wk.as_ref().unwrap(), k_out_size)
            .unwrap();
        println!(
            "[A3] K GEMV ({}->{}): {:.2} ms",
            hidden_size,
            k_out_size,
            t.elapsed().as_secs_f64() * 1000.0
        );

        // V projection
        let v_out_size =
            layer.wv.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);
        let t = Instant::now();
        let v = model
            .gemv_impl(&normed, layer.wv.as_ref().unwrap(), v_out_size)
            .unwrap();
        println!(
            "[A4] V GEMV ({}->{}): {:.2} ms",
            hidden_size,
            v_out_size,
            t.elapsed().as_secs_f64() * 1000.0
        );

        // RoPE Q
        let n_heads = model.config.n_heads;
        let n_kv_heads = model.config.n_kv_heads;
        let head_dim = model.config.head_dim;
        let pos = 0;
        let t = Instant::now();
        let q_rope = model.apply_rope(&q, pos, n_heads, head_dim).unwrap();
        println!("[A5] RoPE Q: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

        // RoPE K
        let t = Instant::now();
        let k_rope = model.apply_rope(&k, pos, n_kv_heads, head_dim).unwrap();
        println!("[A6] RoPE K: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

        // Attention
        let t = Instant::now();
        let attn_out = model
            .attention(
                &q_rope,
                &k_rope,
                &v,
                layer_idx,
                &mut kv_cache,
                n_heads,
                head_dim,
                n_kv_heads,
            )
            .unwrap();
        println!(
            "[A7] Attention: {:.2} ms, output: {}",
            t.elapsed().as_secs_f64() * 1000.0,
            attn_out.len()
        );

        // Output projection
        let t = Instant::now();
        let attn_proj = model
            .gemv_impl(&attn_out, layer.wo.as_ref().unwrap(), hidden_size)
            .unwrap();
        println!("[A8] Wo GEMV: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

        // Residual
        let t = Instant::now();
        let residual: Vec<f32> =
            hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();
        println!(
            "[A9] Residual Add: {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // FFN norm
        let t = Instant::now();
        let ffn_normed = model
            .apply_rms_norm_impl(
                &residual,
                layer.ffn_norm.as_ref().unwrap(),
                hidden_size,
            )
            .unwrap();
        println!("[F1] FFN Norm: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

        // Gate projection
        let intermediate_size =
            layer.w1.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size * 4);
        let t = Instant::now();
        let gate = model
            .gemv_impl(
                &ffn_normed,
                layer.w1.as_ref().unwrap(),
                intermediate_size,
            )
            .unwrap();
        println!(
            "[F2] Gate GEMV ({}->{}): {:.2} ms",
            hidden_size,
            intermediate_size,
            t.elapsed().as_secs_f64() * 1000.0
        );

        // Up projection
        let t = Instant::now();
        let up = model
            .gemv_impl(
                &ffn_normed,
                layer.w3.as_ref().unwrap(),
                intermediate_size,
            )
            .unwrap();
        println!(
            "[F3] Up GEMV ({}->{}): {:.2} ms",
            hidden_size,
            intermediate_size,
            t.elapsed().as_secs_f64() * 1000.0
        );

        // SiLU
        let t = Instant::now();
        let silu: Vec<f32> =
            gate.iter().map(|g| g / (1.0 + (-g).exp())).collect();
        println!("[F4] SiLU: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

        // Mul
        let t = Instant::now();
        let ffn_hidden: Vec<f32> =
            silu.iter().zip(up.iter()).map(|(s, u)| s * u).collect();
        println!("[F5] Mul: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

        // Down projection
        let t = Instant::now();
        let ffn_out = model
            .gemv_impl(&ffn_hidden, layer.w2.as_ref().unwrap(), hidden_size)
            .unwrap();
        println!(
            "[F6] Down GEMV ({}->{}): {:.2} ms",
            intermediate_size,
            hidden_size,
            t.elapsed().as_secs_f64() * 1000.0
        );

        // Final residual
        let t = Instant::now();
        hidden =
            residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect();
        println!(
            "[F7] Residual Add: {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }

    // Final norm
    println!("\n--- Output ---");
    let t = Instant::now();
    let final_normed = model
        .apply_rms_norm_impl(&hidden, model.norm.as_ref().unwrap(), hidden_size)
        .unwrap();
    println!("[1] Final Norm: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

    // Output projection
    let vocab_size = model.config.vocab_size;
    let t = Instant::now();
    let logits = model
        .gemv_impl(&final_normed, model.output.as_ref().unwrap(), vocab_size)
        .unwrap();
    println!(
        "[2] Output GEMV ({}->{}): {:.2} ms",
        hidden_size,
        vocab_size,
        t.elapsed().as_secs_f64() * 1000.0
    );

    println!("\nTotal logits: {}", logits.len());
}
