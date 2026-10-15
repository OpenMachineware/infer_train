use infer_train::model::{GgufFile, LlamaModel, ModelConfig, WeightStore};

fn main() {
    let model_path = std::env::args().nth(1).expect("Need model path");

    let gguf = GgufFile::open(&model_path).expect("Failed to open GGUF");
    let config = ModelConfig::from_gguf(&gguf);
    let store =
        WeightStore::from_gguf(&model_path).expect("Failed to load weights");
    let model = LlamaModel::from_weights(&store, config);

    println!("=== MLA Config ===");
    println!("kv_lora_rank: {}", model.config.kv_lora_rank);
    println!("qk_rope_head_dim: {}", model.config.qk_rope_head_dim);
    println!("key_length: {}", model.config.key_length);
    println!("value_length: {}", model.config.value_length);
    println!("n_heads: {}", model.config.n_heads);
    println!("n_kv_heads: {}", model.config.n_kv_heads);
    println!("head_dim (config): {}", model.config.head_dim);

    let layer = &model.layers[0];
    println!("\n=== Layer 0 MQA Weights ===");
    if let Some(wq) = &layer.wq {
        println!("wq: shape={:?}, dtype={:?}", wq.shape, wq.dtype);
        println!(
            "  => Q head_dim = {} / {} = {}",
            wq.shape[1],
            model.config.n_heads,
            wq.shape[1] / model.config.n_heads
        );
    }
    if let Some(kv_a) = &layer.attn_kv_a_mqa {
        println!(
            "attn_kv_a_mqa: shape={:?}, dtype={:?}",
            kv_a.shape, kv_a.dtype
        );
        println!(
            "  => Expected kv_lora_rank + qk_rope_head_dim = {} + {} = {}",
            model.config.kv_lora_rank,
            model.config.qk_rope_head_dim,
            model.config.kv_lora_rank + model.config.qk_rope_head_dim
        );
    }
    if let Some(kv_a_norm) = &layer.attn_kv_a_norm {
        println!(
            "attn_kv_a_norm: shape={:?}, dtype={:?}",
            kv_a_norm.shape, kv_a_norm.dtype
        );
    }
    if let Some(kv_b) = &layer.attn_kv_b {
        println!("attn_kv_b: shape={:?}, dtype={:?}", kv_b.shape, kv_b.dtype);
        let qk_nope_head_dim =
            model.config.key_length - model.config.qk_rope_head_dim;
        let expected_kv_b_out = model.config.n_kv_heads
            * (qk_nope_head_dim + model.config.value_length);
        println!(
            "  => Expected out = n_kv_heads * (qk_nope + value) = {} * ({} + {}) = {}",
            model.config.n_kv_heads,
            qk_nope_head_dim,
            model.config.value_length,
            expected_kv_b_out
        );
    }
    if let Some(wo) = &layer.wo {
        println!("wo: shape={:?}, dtype={:?}", wo.shape, wo.dtype);
    }

    // Test with a single token
    println!("\n=== Test Forward ===");
    let token = 100000u32; // BOS
    let hidden = model.embedding_lookup_impl(token).expect("Embedding failed");
    println!(
        "Embedding: len={}, range=[{:.4}, {:.4}]",
        hidden.len(),
        hidden.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Test attention norm
    let normed = model
        .apply_rms_norm_impl(
            &hidden,
            layer.attention_norm.as_ref().unwrap(),
            hidden.len(),
        )
        .unwrap();
    println!(
        "After norm: len={}, range=[{:.4}, {:.4}]",
        normed.len(),
        normed.iter().cloned().fold(f32::INFINITY, f32::min),
        normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Test QKV projection
    println!("\n=== QKV Projection ===");
    let (q, k, v, q_size, k_size, v_size) = model
        .get_qkv_proj(
            layer,
            &normed,
            model.config.n_heads,
            model.config.n_kv_heads,
            model.config.head_dim,
        )
        .expect("QKV projection failed");

    println!(
        "Q: len={} (expected {} * {} = {})",
        q.len(),
        model.config.n_heads,
        q.len() / model.config.n_heads,
        q.len()
    );
    println!(
        "K: len={} (expected {} * {} = {})",
        k.len(),
        model.config.n_kv_heads,
        model.config.key_length,
        model.config.n_kv_heads * model.config.key_length
    );
    println!(
        "V: len={} (expected {} * {} = {})",
        v.len(),
        model.config.n_kv_heads,
        model.config.value_length,
        model.config.n_kv_heads * model.config.value_length
    );

    println!(
        "\nQ range: [{:.4}, {:.4}]",
        q.iter().cloned().fold(f32::INFINITY, f32::min),
        q.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "K range: [{:.4}, {:.4}]",
        k.iter().cloned().fold(f32::INFINITY, f32::min),
        k.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "V range: [{:.4}, {:.4}]",
        v.iter().cloned().fold(f32::INFINITY, f32::min),
        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Check for NaN
    let q_nan = q.iter().filter(|&&x| x.is_nan()).count();
    let k_nan = k.iter().filter(|&&x| x.is_nan()).count();
    let v_nan = v.iter().filter(|&&x| x.is_nan()).count();
    println!("\nNaN count: Q={}, K={}, V={}", q_nan, k_nan, v_nan);

    // Check Layer 1 MoE weights
    println!("\n=== Layer 1 MoE Weights ===");
    if model.layers.len() > 1 {
        let layer1 = &model.layers[1];
        println!("attn_norm: {}", layer1.attention_norm.is_some());
        println!("ffn_norm: {}", layer1.ffn_norm.is_some());
        println!("ffn_gate_inp: {}", layer1.ffn_gate_inp.is_some());
        println!("ffn_gate_exps: {}", layer1.ffn_gate_exps.is_some());
        println!("ffn_up_exps: {}", layer1.ffn_up_exps.is_some());
        println!("ffn_down_exps: {}", layer1.ffn_down_exps.is_some());
        println!("ffn_gate_shexp: {}", layer1.ffn_gate_shexp.is_some());
        println!("ffn_up_shexp: {}", layer1.ffn_up_shexp.is_some());
        println!("ffn_down_shexp: {}", layer1.ffn_down_shexp.is_some());

        if let Some(gate_inp) = &layer1.ffn_gate_inp {
            println!(
                "\nffn_gate_inp: shape={:?}, dtype={:?}",
                gate_inp.shape, gate_inp.dtype
            );
        }
        if let Some(gate_exps) = &layer1.ffn_gate_exps {
            println!(
                "ffn_gate_exps: shape={:?}, dtype={:?}",
                gate_exps.shape, gate_exps.dtype
            );
        }
        if let Some(up_exps) = &layer1.ffn_up_exps {
            println!(
                "ffn_up_exps: shape={:?}, dtype={:?}",
                up_exps.shape, up_exps.dtype
            );
        }
        if let Some(down_exps) = &layer1.ffn_down_exps {
            println!(
                "ffn_down_exps: shape={:?}, dtype={:?}",
                down_exps.shape, down_exps.dtype
            );
        }
        if let Some(gate_shexp) = &layer1.ffn_gate_shexp {
            println!(
                "ffn_gate_shexp: shape={:?}, dtype={:?}",
                gate_shexp.shape, gate_shexp.dtype
            );
        }
        if let Some(up_shexp) = &layer1.ffn_up_shexp {
            println!(
                "ffn_up_shexp: shape={:?}, dtype={:?}",
                up_shexp.shape, up_shexp.dtype
            );
        }
        if let Some(down_shexp) = &layer1.ffn_down_shexp {
            println!(
                "ffn_down_shexp: shape={:?}, dtype={:?}",
                down_shexp.shape, down_shexp.dtype
            );
        }
    }
}
