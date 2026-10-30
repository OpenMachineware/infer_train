// Profile CPU forward pass with detailed timing - enhanced version

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
    println!("=== CPU Detailed Profile v2 ===\n");

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!(
        "Model: {} layers, hidden={}, intermediate={}",
        ctx.model.layers.len(),
        ctx.config.hidden_size,
        ctx.config.intermediate_size
    );

    let hidden_size = ctx.config.hidden_size;
    let intermediate_size = ctx.config.intermediate_size;
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;
    let n_layers = ctx.model.layers.len();
    let vocab_size = ctx.config.vocab_size;

    let qkv_out_size =
        ctx.model.layers[0].attn_qkv.as_ref().map(|t| t.shape[1]).unwrap_or(0);

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
        2048,
        n_layers,
    );
    buffers.init_norm_weights(&ctx.model);

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Warmup
    let _ = ctx.model.forward(&[1, 2, 3], &mut kv_cache);
    kv_cache.clear();

    // Detailed breakdown for first layer
    println!("\n=== First Layer Breakdown ===");
    let layer = &ctx.model.layers[0];

    // Attention norm
    let t = Instant::now();
    let attn_norm_offset = 0;
    ctx.model.apply_rms_norm_predequant(
        &buffers.hidden,
        &buffers.norm_weights[attn_norm_offset..attn_norm_offset + hidden_size],
        &mut buffers.hidden_alt,
    );
    println!("Attn norm: {} µs", t.elapsed().as_micros());

    // QKV projections individually
    let mut total_qkv = 0u64;

    let t = Instant::now();
    if let Some(ref qkv) = layer.attn_qkv {
        ctx.model.gemv_into_with_buffers(
            &buffers.hidden_alt,
            qkv,
            &mut buffers.qkv_buffer[..qkv.shape[1]],
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );
    } else {
        // Separate wq/wk/wv
        let x_q8k_blocks = unsafe {
            use infer_train::quant::types::{BlockQ8K, QK_K};
            let in_size_padded = ((hidden_size + QK_K - 1) / QK_K) * QK_K;
            let nb = in_size_padded / QK_K;
            std::slice::from_raw_parts_mut(
                buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
                nb,
            )
        };
        LlamaModel::quantize_input_to_q8k(
            &buffers.hidden_alt,
            x_q8k_blocks,
            &mut buffers.x_padded,
        );

        let t1 = Instant::now();
        ctx.model.gemv_with_prequant(
            x_q8k_blocks,
            hidden_size,
            layer.wq.as_ref().unwrap(),
            &mut buffers.q_buffer[..n_heads * head_dim],
        );
        let wq_time = t1.elapsed().as_micros();
        println!(
            "wq GEMV ({}x{}): {} µs",
            hidden_size,
            n_heads * head_dim,
            wq_time
        );
        total_qkv += wq_time as u64;

        let t2 = Instant::now();
        ctx.model.gemv_with_prequant(
            x_q8k_blocks,
            hidden_size,
            layer.wk.as_ref().unwrap(),
            &mut buffers.k_buffer[..n_kv_heads * head_dim],
        );
        let wk_time = t2.elapsed().as_micros();
        println!(
            "wk GEMV ({}x{}): {} µs",
            hidden_size,
            n_kv_heads * head_dim,
            wk_time
        );
        total_qkv += wk_time as u64;

        let t3 = Instant::now();
        ctx.model.gemv_with_prequant(
            x_q8k_blocks,
            hidden_size,
            layer.wv.as_ref().unwrap(),
            &mut buffers.v_buffer[..n_kv_heads * head_dim],
        );
        let wv_time = t3.elapsed().as_micros();
        println!(
            "wv GEMV ({}x{}): {} µs",
            hidden_size,
            n_kv_heads * head_dim,
            wv_time
        );
        total_qkv += wv_time as u64;
    }
    let qkv_time = t.elapsed().as_micros();
    if total_qkv > 0 {
        println!(
            "QKV quant + GEMV total: {} µs (quant: {} µs)",
            qkv_time,
            qkv_time as i64 - total_qkv as i64
        );
    } else {
        println!("QKV fused GEMV: {} µs", qkv_time);
    }

    // RoPE
    let t = Instant::now();
    let pos = 0;
    let q_size = n_heads * head_dim;
    ctx.model
        .apply_rope_into(
            &buffers.q_buffer[..q_size],
            pos,
            n_heads,
            head_dim,
            &mut buffers.rope_buffer,
        )
        .unwrap();
    buffers.q_buffer[..q_size].copy_from_slice(&buffers.rope_buffer[..q_size]);
    let k_size = n_kv_heads * head_dim;
    ctx.model
        .apply_rope_into(
            &buffers.k_buffer[..k_size],
            pos,
            n_kv_heads,
            head_dim,
            &mut buffers.rope_buffer,
        )
        .unwrap();
    buffers.k_buffer[..k_size].copy_from_slice(&buffers.rope_buffer[..k_size]);
    println!("RoPE: {} µs", t.elapsed().as_micros());

    // Attention
    let t = Instant::now();
    ctx.model
        .attention_zero_alloc(
            &buffers.q_buffer,
            &buffers.k_buffer,
            &buffers.v_buffer,
            0,
            &mut kv_cache,
            n_heads,
            head_dim,
            n_kv_heads,
            &mut buffers.attn_out,
            &mut buffers.attn_scores,
        )
        .unwrap();
    println!("Attention: {} µs", t.elapsed().as_micros());

    // wo projection
    let t = Instant::now();
    ctx.model.gemv_into_with_buffers(
        &buffers.attn_out,
        layer.wo.as_ref().unwrap(),
        &mut buffers.hidden_alt,
        &mut buffers.q8k_input,
        &mut buffers.x_padded,
    );
    println!(
        "wo GEMV ({}x{}): {} µs",
        hidden_size,
        hidden_size,
        t.elapsed().as_micros()
    );

    // Residual
    for i in 0..hidden_size {
        buffers.hidden[i] += buffers.hidden_alt[i];
    }

    // FFN norm
    let t = Instant::now();
    let ffn_norm_offset = hidden_size;
    ctx.model.apply_rms_norm_predequant(
        &buffers.hidden,
        &buffers.norm_weights[ffn_norm_offset..ffn_norm_offset + hidden_size],
        &mut buffers.hidden_alt,
    );
    println!("FFN norm: {} µs", t.elapsed().as_micros());

    // FFN breakdown
    println!("\nFFN Breakdown:");
    let x_q8k_blocks = unsafe {
        use infer_train::quant::types::{BlockQ8K, QK_K};
        let in_size_padded = ((hidden_size + QK_K - 1) / QK_K) * QK_K;
        let nb = in_size_padded / QK_K;
        std::slice::from_raw_parts_mut(
            buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
            nb,
        )
    };
    LlamaModel::quantize_input_to_q8k(
        &buffers.hidden_alt,
        x_q8k_blocks,
        &mut buffers.x_padded,
    );

    let t = Instant::now();
    ctx.model.gemv_with_prequant(
        x_q8k_blocks,
        hidden_size,
        layer.w1.as_ref().unwrap(),
        &mut buffers.ffn_gate,
    );
    println!(
        "w1 GEMV ({}x{}): {} µs",
        hidden_size,
        intermediate_size,
        t.elapsed().as_micros()
    );

    let t = Instant::now();
    ctx.model.gemv_with_prequant(
        x_q8k_blocks,
        hidden_size,
        layer.w3.as_ref().unwrap(),
        &mut buffers.ffn_up,
    );
    println!(
        "w3 GEMV ({}x{}): {} µs",
        hidden_size,
        intermediate_size,
        t.elapsed().as_micros()
    );

    // SiLU + mul
    let t = Instant::now();
    for i in 0..intermediate_size {
        let silu = buffers.ffn_gate[i] / (1.0 + buffers.ffn_gate[i].exp());
        buffers.ffn_up[i] = silu * buffers.ffn_up[i];
    }
    println!("SiLU+mul: {} µs", t.elapsed().as_micros());

    let t = Instant::now();
    ctx.model.gemv_with_prequant(
        x_q8k_blocks,
        hidden_size,
        layer.w2.as_ref().unwrap(),
        &mut buffers.hidden,
    );
    println!(
        "w2 GEMV ({}x{}): {} µs",
        intermediate_size,
        hidden_size,
        t.elapsed().as_micros()
    );

    // Full token benchmark
    println!("\n=== Full Token Benchmark ===");
    kv_cache.clear();
    buffers.hidden.fill(0.0);
    buffers.hidden_alt.fill(0.0);

    let start = Instant::now();
    for _ in 0..5 {
        let mut total_gemv = 0u64;
        let mut total_attn = 0u64;
        let mut total_norm = 0u64;
        let mut total_ffn = 0u64;

        for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
            let t1 = Instant::now();
            let attn_norm_offset = layer_idx * 2 * hidden_size;
            ctx.model.apply_rms_norm_predequant(
                &buffers.hidden,
                &buffers.norm_weights
                    [attn_norm_offset..attn_norm_offset + hidden_size],
                &mut buffers.hidden_alt,
            );
            total_norm += t1.elapsed().as_micros() as u64;

            let t2 = Instant::now();
            if let Some(ref qkv) = layer.attn_qkv {
                // Use single-thread GEMV for decode
                ctx.model.gemv_into_with_buffers_single_thread(
                    &buffers.hidden_alt,
                    qkv,
                    &mut buffers.qkv_buffer[..qkv.shape[1]],
                    &mut buffers.q8k_input,
                    &mut buffers.x_padded,
                );
            } else {
                let x_q8k_blocks = unsafe {
                    use infer_train::quant::types::{BlockQ8K, QK_K};
                    let in_size_padded =
                        ((hidden_size + QK_K - 1) / QK_K) * QK_K;
                    let nb = in_size_padded / QK_K;
                    std::slice::from_raw_parts_mut(
                        buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
                        nb,
                    )
                };
                LlamaModel::quantize_input_to_q8k(
                    &buffers.hidden_alt,
                    x_q8k_blocks,
                    &mut buffers.x_padded,
                );
                // Use single-thread GEMV for decode (eliminates thread creation overhead)
                ctx.model.gemv_with_prequant_single_thread(
                    x_q8k_blocks,
                    hidden_size,
                    layer.wq.as_ref().unwrap(),
                    &mut buffers.q_buffer[..n_heads * head_dim],
                );
                ctx.model.gemv_with_prequant_single_thread(
                    x_q8k_blocks,
                    hidden_size,
                    layer.wk.as_ref().unwrap(),
                    &mut buffers.k_buffer[..n_kv_heads * head_dim],
                );
                ctx.model.gemv_with_prequant_single_thread(
                    x_q8k_blocks,
                    hidden_size,
                    layer.wv.as_ref().unwrap(),
                    &mut buffers.v_buffer[..n_kv_heads * head_dim],
                );
            }

            let pos = kv_cache.seq_len();
            let q_size = n_heads * head_dim;
            ctx.model
                .apply_rope_into(
                    &buffers.q_buffer[..q_size],
                    pos,
                    n_heads,
                    head_dim,
                    &mut buffers.rope_buffer,
                )
                .unwrap();
            buffers.q_buffer[..q_size]
                .copy_from_slice(&buffers.rope_buffer[..q_size]);

            let k_size = n_kv_heads * head_dim;
            ctx.model
                .apply_rope_into(
                    &buffers.k_buffer[..k_size],
                    pos,
                    n_kv_heads,
                    head_dim,
                    &mut buffers.rope_buffer,
                )
                .unwrap();
            buffers.k_buffer[..k_size]
                .copy_from_slice(&buffers.rope_buffer[..k_size]);

            ctx.model
                .attention_zero_alloc(
                    &buffers.q_buffer,
                    &buffers.k_buffer,
                    &buffers.v_buffer,
                    layer_idx,
                    &mut kv_cache,
                    n_heads,
                    head_dim,
                    n_kv_heads,
                    &mut buffers.attn_out,
                    &mut buffers.attn_scores,
                )
                .unwrap();

            // Use single-thread GEMV for wo projection
            ctx.model.gemv_into_with_buffers_single_thread(
                &buffers.attn_out,
                layer.wo.as_ref().unwrap(),
                &mut buffers.hidden_alt,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
            total_gemv += t2.elapsed().as_micros() as u64;

            for i in 0..hidden_size {
                buffers.hidden[i] += buffers.hidden_alt[i];
            }

            let t3 = Instant::now();
            let ffn_norm_offset = layer_idx * 2 * hidden_size + hidden_size;
            ctx.model.apply_rms_norm_predequant(
                &buffers.hidden,
                &buffers.norm_weights
                    [ffn_norm_offset..ffn_norm_offset + hidden_size],
                &mut buffers.hidden_alt,
            );

            // For FFN: use multi-thread for large matrices (compute > thread overhead)
            // w1 and w3: 1536x8960 (large, use multi-thread)
            // w2: 8960x1536 (large, use multi-thread)
            ctx.model
                .compute_ffn_zero_alloc(
                    layer,
                    &buffers.hidden_alt,
                    &mut buffers.ffn_gate,
                    &mut buffers.ffn_up,
                    &mut buffers.hidden,
                    &mut buffers.q8k_input,
                    &mut buffers.x_padded,
                )
                .unwrap();
            total_ffn += t3.elapsed().as_micros() as u64;
        }

        kv_cache.advance();

        println!(
            "GEMV: {} µs, Attn: {} µs, Norm: {} µs, FFN: {} µs",
            total_gemv, total_attn, total_norm, total_ffn
        );
    }

    let elapsed = start.elapsed();
    println!("\nTotal 5 tokens: {:?}", elapsed);
    println!("Per token: {} µs", elapsed.as_micros() / 5);
    println!("Speed: {:.1} t/s", 5.0 / elapsed.as_secs_f64());

    println!("\n=== Memory Bandwidth Estimate ===");
    // For 1.5B model: ~1.5B parameters
    // Q5_K: ~0.7 bytes per parameter
    // Total weight size: ~1GB
    // Per token: all weights touched once
    let weight_size_gb = 1.5 * 0.7 / 1.0; // 1.05 GB
    let time_s = (elapsed.as_micros() / 5) as f64 / 1_000_000.0;
    let bandwidth_gbs = weight_size_gb / time_s;
    println!("Weight size: {:.2} GB", weight_size_gb);
    println!("Time per token: {:.3} ms", time_s * 1000.0);
    println!("Memory bandwidth: {:.1} GB/s", bandwidth_gbs);
}
