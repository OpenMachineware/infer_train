// Measure actual decode timing breakdown
// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::KVCacheBase;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");

    let hidden_size = ctx.config.hidden_size;
    let vocab_size = ctx.config.vocab_size;

    println!("=== Actual Decode Timing ===");
    println!("hidden_size: {}, vocab_size: {}", hidden_size, vocab_size);

    // Create buffers
    let mut kv = infer_train::inference::KVCache::new(
        &ctx.config,
        2048,
        infer_train::inference::KVCacheFormat::F32,
    );

    let mut buffer_pool = infer_train::cpu::CpuWorkBuffersPool::new(&ctx.model);
    buffer_pool.preallocate(1);
    let mut buffers = buffer_pool.allocate().unwrap();

    // Prefill with single token
    let _ = ctx.forward_pooled(&[1], &mut kv);

    // Measure decode
    let mut total_times = Vec::new();
    let mut layer_times = Vec::new();
    let mut output_times = Vec::new();
    let mut qkv_times = Vec::new();
    let mut attn_times = Vec::new();
    let mut wo_times = Vec::new();
    let mut ffn_times = Vec::new();
    let mut other_times = Vec::new();

    for i in 0..20 {
        let total_start = Instant::now();

        // Embedding lookup
        ctx.model.embedding_lookup_into(100, &mut buffers.hidden);

        // Layers - detailed breakdown
        let mut total_qkv = 0u128;
        let mut total_attn = 0u128;
        let mut total_wo = 0u128;
        let mut total_ffn = 0u128;
        let mut total_other = 0u128;

        let layer_start = Instant::now();
        for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
            // QKV projection
            let t1 = Instant::now();
            ctx.model.apply_rms_norm_predequant(
                &buffers.hidden,
                &buffers.norm_weights[layer_idx * 2 * hidden_size..]
                    [..hidden_size],
                &mut buffers.hidden_alt,
            );
            if let Some(ref qkv) = layer.attn_qkv {
                ctx.model.gemv_into_with_buffers_single_thread(
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
                    let nb = ((hidden_size + QK_K - 1) / QK_K) * QK_K / QK_K;
                    std::slice::from_raw_parts_mut(
                        buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
                        nb,
                    )
                };
                infer_train::model::LlamaModel::quantize_input_to_q8k(
                    &buffers.hidden_alt,
                    x_q8k_blocks,
                    &mut buffers.x_padded,
                );
                ctx.model.gemv_with_prequant_single_thread(
                    x_q8k_blocks,
                    hidden_size,
                    layer.wq.as_ref().unwrap(),
                    &mut buffers.q_buffer[..],
                );
                ctx.model.gemv_with_prequant_single_thread(
                    x_q8k_blocks,
                    hidden_size,
                    layer.wk.as_ref().unwrap(),
                    &mut buffers.k_buffer[..],
                );
                ctx.model.gemv_with_prequant_single_thread(
                    x_q8k_blocks,
                    hidden_size,
                    layer.wv.as_ref().unwrap(),
                    &mut buffers.v_buffer[..],
                );
            }
            total_qkv += t1.elapsed().as_nanos();

            // Attention
            let t2 = Instant::now();
            let pos = kv.seq_len();
            let q_size = ctx.config.n_heads * ctx.config.head_dim;
            ctx.model
                .apply_rope_into(
                    &buffers.q_buffer[..q_size],
                    pos,
                    ctx.config.n_heads,
                    ctx.config.head_dim,
                    &mut buffers.rope_buffer,
                )
                .unwrap();
            buffers.q_buffer[..q_size]
                .copy_from_slice(&buffers.rope_buffer[..q_size]);

            let k_size = ctx.config.n_kv_heads * ctx.config.head_dim;
            ctx.model
                .apply_rope_into(
                    &buffers.k_buffer[..k_size],
                    pos,
                    ctx.config.n_kv_heads,
                    ctx.config.head_dim,
                    &mut buffers.rope_buffer,
                )
                .unwrap();
            buffers.k_buffer[..k_size]
                .copy_from_slice(&buffers.rope_buffer[..k_size]);

            // Flash attention
            #[cfg(target_arch = "aarch64")]
            {
                infer_train::cpu::flash_attention_single(
                    &buffers.q_buffer[..q_size],
                    &buffers.k_buffer[..k_size],
                    &buffers.v_buffer[..],
                    layer_idx,
                    &mut kv,
                    ctx.config.n_heads,
                    ctx.config.head_dim,
                    ctx.config.n_kv_heads,
                    &mut buffers.attn_out,
                    &mut buffers.flash_attn_prefill,
                );
            }
            total_attn += t2.elapsed().as_nanos();

            // Output projection
            let t3 = Instant::now();
            ctx.model.gemv_into_with_buffers_single_thread(
                &buffers.attn_out,
                layer.wo.as_ref().unwrap(),
                &mut buffers.hidden_alt,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
            // Residual add
            for j in 0..hidden_size {
                buffers.hidden[j] += buffers.hidden_alt[j];
            }
            total_wo += t3.elapsed().as_nanos();

            // FFN
            let t4 = Instant::now();
            // FFN norm
            let ffn_norm_offset = layer_idx * 2 * hidden_size + hidden_size;
            ctx.model.apply_rms_norm_predequant(
                &buffers.hidden,
                &buffers.norm_weights[ffn_norm_offset..][..hidden_size],
                &mut buffers.hidden_alt,
            );
            ctx.model
                .compute_ffn_zero_alloc_single_thread(
                    layer,
                    &buffers.hidden_alt,
                    &mut buffers.ffn_gate,
                    &mut buffers.ffn_up,
                    &mut buffers.hidden,
                    &mut buffers.q8k_input,
                    &mut buffers.x_padded,
                )
                .unwrap();
            total_ffn += t4.elapsed().as_nanos();
        }
        let layer_time = layer_start.elapsed();

        // Final norm + output projection
        let output_start = Instant::now();
        ctx.model.apply_rms_norm_predequant(
            &buffers.hidden,
            &buffers.norm_weight,
            &mut buffers.hidden_alt,
        );
        ctx.model.gemv_into_with_buffers_single_thread(
            &buffers.hidden_alt,
            ctx.model.output.as_ref().unwrap(),
            &mut buffers.logits[..vocab_size],
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );
        let output_time = output_start.elapsed();

        kv.advance();
        let total_time = total_start.elapsed();

        if i >= 5 {
            total_times.push(total_time);
            layer_times.push(layer_time);
            output_times.push(output_time);
            qkv_times.push(total_qkv);
            attn_times.push(total_attn);
            wo_times.push(total_wo);
            ffn_times.push(total_ffn);
            other_times.push(total_other);
        }
    }

    let avg_total: std::time::Duration =
        total_times.iter().sum::<std::time::Duration>()
            / total_times.len() as u32;
    let avg_layer: std::time::Duration =
        layer_times.iter().sum::<std::time::Duration>()
            / layer_times.len() as u32;
    let avg_output: std::time::Duration =
        output_times.iter().sum::<std::time::Duration>()
            / output_times.len() as u32;
    let avg_qkv: u128 =
        qkv_times.iter().sum::<u128>() / qkv_times.len() as u128;
    let avg_attn: u128 =
        attn_times.iter().sum::<u128>() / attn_times.len() as u128;
    let avg_wo: u128 = wo_times.iter().sum::<u128>() / wo_times.len() as u128;
    let avg_ffn: u128 =
        ffn_times.iter().sum::<u128>() / ffn_times.len() as u128;

    println!("\n=== Timing Breakdown (15 tokens) ===");
    println!("Total decode: {:?}", avg_total);
    println!("Layers: {:?}", avg_layer);
    println!("  QKV (norm+quant+gemv): {:.2} ms", avg_qkv as f64 / 1_000_000.0);
    println!(
        "  Attention (rope+flash): {:.2} ms",
        avg_attn as f64 / 1_000_000.0
    );
    println!("  wo + residual: {:.2} ms", avg_wo as f64 / 1_000_000.0);
    println!("  FFN: {:.2} ms", avg_ffn as f64 / 1_000_000.0);
    println!("Output projection + norm: {:?}", avg_output);

    let layer_sum = avg_qkv + avg_attn + avg_wo + avg_ffn;
    println!(
        "\nLayer breakdown total: {:.2} ms",
        layer_sum as f64 / 1_000_000.0
    );
    println!(
        "Unaccounted: {:?}",
        avg_layer - std::time::Duration::from_nanos(layer_sum as u64)
    );

    println!("\nPercentage:");
    println!(
        "  Layers: {:.1}%",
        avg_layer.as_secs_f64() / avg_total.as_secs_f64() * 100.0
    );
    println!(
        "  Output: {:.1}%",
        avg_output.as_secs_f64() / avg_total.as_secs_f64() * 100.0
    );

    println!("\nSpeed: {:.1} t/s", 1.0 / avg_total.as_secs_f64());
}
