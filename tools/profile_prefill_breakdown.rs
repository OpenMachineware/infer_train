// Detailed prefill profiling
use infer_train::{
    cpu::CpuWorkBuffers,
    inference::{
        Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
    },
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let mut ctx = InferenceContext::from_gguf(&args[1], Device::Cpu)
        .expect("Failed to load model");
    let mut buffers = CpuWorkBuffers::new(
        ctx.config.hidden_size,
        ctx.config.intermediate_size,
        ctx.config.n_heads,
        ctx.config.n_kv_heads,
        ctx.config.head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        ctx.config.vocab_size,
        ctx.model
            .layers
            .first()
            .and_then(|l| l.attn_qkv.as_ref())
            .map(|t| t.shape[1])
            .unwrap_or(0),
        2048,
        ctx.model.layers.len(),
    );
    buffers.init_norm_weights(&ctx.model);

    let n_tokens = 64;
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let tokens: Vec<u32> = (1..=n_tokens as u32).collect();

    // Embedding
    let mut batch_hidden = vec![0.0f32; n_tokens * ctx.config.hidden_size];
    for (i, &token) in tokens.iter().enumerate() {
        ctx.model.embedding_lookup_into(
            token,
            &mut batch_hidden[i * ctx.config.hidden_size..],
        );
    }

    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;
    let n_heads = ctx.config.n_heads;
    let head_dim = ctx.config.head_dim;
    let n_kv_heads = ctx.config.n_kv_heads;

    // Norm
    let start = Instant::now();
    for i in 0..n_tokens {
        ctx.model.apply_rms_norm_predequant(
            &batch_hidden[i * hidden_size..(i + 1) * hidden_size],
            &buffers.norm_weights[0..hidden_size],
            &mut buffers.hidden_alt,
        );
        batch_hidden[i * hidden_size..(i + 1) * hidden_size]
            .copy_from_slice(&buffers.hidden_alt);
    }
    let norm_time = start.elapsed();

    // QKV GEMMs
    let inputs: Vec<&[f32]> = (0..n_tokens)
        .map(|i| &batch_hidden[i * hidden_size..(i + 1) * hidden_size])
        .collect();

    let start = Instant::now();
    let q_size = n_heads * head_dim;
    let k_size = n_kv_heads * head_dim;
    let v_size = n_kv_heads * head_dim;

    let mut q_outputs = vec![0.0f32; n_tokens * q_size];
    let mut k_outputs = vec![0.0f32; n_tokens * k_size];
    let mut v_outputs = vec![0.0f32; n_tokens * v_size];

    if let Some(ref wq) = layer.wq {
        ctx.model.gemm_batch_prefill(
            &inputs,
            wq,
            &mut q_outputs,
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );
    }
    if let Some(ref wk) = layer.wk {
        ctx.model.gemm_batch_prefill(
            &inputs,
            wk,
            &mut k_outputs,
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );
    }
    if let Some(ref wv) = layer.wv {
        ctx.model.gemm_batch_prefill(
            &inputs,
            wv,
            &mut v_outputs,
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );
    }
    let qkv_time = start.elapsed();

    // Sequential attention (RoPE + attention + wo)
    let start = Instant::now();
    for (token_idx, pos) in (0..n_tokens).zip(kv.seq_len()..) {
        buffers.q_buffer[..q_size].copy_from_slice(
            &q_outputs[token_idx * q_size..(token_idx + 1) * q_size],
        );
        buffers.k_buffer[..k_size].copy_from_slice(
            &k_outputs[token_idx * k_size..(token_idx + 1) * k_size],
        );
        buffers.v_buffer[..v_size].copy_from_slice(
            &v_outputs[token_idx * v_size..(token_idx + 1) * v_size],
        );

        ctx.model
            .apply_rope_into(
                &buffers.q_buffer,
                pos,
                n_heads,
                head_dim,
                &mut buffers.rope_buffer,
            )
            .unwrap();
        buffers.q_buffer[..q_size]
            .copy_from_slice(&buffers.rope_buffer[..q_size]);

        ctx.model
            .apply_rope_into(
                &buffers.k_buffer,
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
                0,
                &mut kv,
                n_heads,
                head_dim,
                n_kv_heads,
                &mut buffers.attn_out,
                &mut buffers.attn_scores,
            )
            .unwrap();

        if let Some(ref wo) = layer.wo {
            ctx.model.gemv_into_with_buffers(
                &buffers.attn_out,
                wo,
                &mut batch_hidden[token_idx * hidden_size..],
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
        }
        kv.advance();
    }
    let attention_time = start.elapsed();

    // FFN
    let start = Instant::now();
    let ffn_inputs: Vec<&[f32]> = (0..n_tokens)
        .map(|i| &batch_hidden[i * hidden_size..(i + 1) * hidden_size])
        .collect();

    if let Some(ref w1) = layer.w1 {
        let intermediate_size = w1.shape[0];
        let mut gate_outputs = vec![0.0f32; n_tokens * intermediate_size];
        ctx.model.gemm_batch_prefill(
            &ffn_inputs,
            w1,
            &mut gate_outputs,
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );

        let mut up_outputs = vec![0.0f32; n_tokens * intermediate_size];
        if let Some(ref w3) = layer.w3 {
            ctx.model.gemm_batch_prefill(
                &ffn_inputs,
                w3,
                &mut up_outputs,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
        }

        // SiLU activation
        for i in 0..gate_outputs.len() {
            gate_outputs[i] = (1.0 / (1.0 + (-gate_outputs[i]).exp()))
                * gate_outputs[i]
                * up_outputs[i];
        }

        // Down projection
        if let Some(ref w2) = layer.w2 {
            let down_inputs: Vec<&[f32]> = (0..n_tokens)
                .map(|i| {
                    &gate_outputs
                        [i * intermediate_size..(i + 1) * intermediate_size]
                })
                .collect();
            ctx.model.gemm_batch_prefill(
                &down_inputs,
                w2,
                &mut batch_hidden,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
        }
    }
    let ffn_time = start.elapsed();

    println!("\n=== Detailed Prefill Timing (64 tokens, 1 layer) ===");
    println!("Norm:         {:?}", norm_time);
    println!("QKV GEMM:     {:?}", qkv_time);
    println!("Attention:    {:?}", attention_time);
    println!("  per token:  {:?}", attention_time / n_tokens as u32);
    println!("FFN:          {:?}", ffn_time);

    let total_layer = norm_time + qkv_time + attention_time + ffn_time;
    println!("\nTotal layer:  {:?}", total_layer);
    println!("Estimated total (28 layers): {:?}", total_layer * 28);

    println!("\n--- Actual prefill ---");
    let mut kv2 = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let start = Instant::now();
    let _ = ctx.forward_pooled(&tokens, &mut kv2);
    let actual = start.elapsed();
    println!("Actual:       {:?}", actual);
    println!("t/s:          {:.2}", n_tokens as f64 / actual.as_secs_f64());
}
