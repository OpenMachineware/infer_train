use infer_train::cpu::CpuWorkBuffersPool;
use infer_train::inference::{
    InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let mut ctx = InferenceContext::from_gguf(
        &args[1],
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");

    let hidden_size = ctx.config.hidden_size;
    let vocab_size = ctx.config.vocab_size;

    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let mut buffer_pool = CpuWorkBuffersPool::new(&ctx.model);
    buffer_pool.preallocate(1);
    let mut buffers = buffer_pool.allocate().unwrap();

    let _ = ctx.forward_pooled(&[1], &mut kv);

    let mut times = Vec::new();

    for i in 0..20 {
        let total_start = Instant::now();

        // Embedding
        ctx.model.embedding_lookup_into(100, &mut buffers.hidden);

        // Layers - NO intermediate timing
        for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
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
                // Extract Q, K, V
                let q_size = ctx.config.n_heads * ctx.config.head_dim;
                let k_size = ctx.config.n_kv_heads * ctx.config.key_length;
                let v_size = ctx.config.n_kv_heads * ctx.config.value_length;
                buffers.q_buffer[..q_size]
                    .copy_from_slice(&buffers.qkv_buffer[..q_size]);
                buffers.k_buffer[..k_size].copy_from_slice(
                    &buffers.qkv_buffer[q_size..q_size + k_size],
                );
                buffers.v_buffer[..v_size].copy_from_slice(
                    &buffers.qkv_buffer
                        [q_size + k_size..q_size + k_size + v_size],
                );

                // RoPE
                let pos = kv.seq_len();
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
                infer_train::cpu::flash_attention_single(
                    &buffers.q_buffer[..q_size],
                    &buffers.k_buffer[..k_size],
                    &buffers.v_buffer[..v_size],
                    layer_idx,
                    &mut kv,
                    ctx.config.n_heads,
                    ctx.config.head_dim,
                    ctx.config.n_kv_heads,
                    &mut buffers.attn_out,
                    &mut buffers.flash_attn_prefill,
                );
            }

            // wo
            ctx.model.gemv_into_with_buffers_single_thread(
                &buffers.attn_out,
                layer.wo.as_ref().unwrap(),
                &mut buffers.hidden_alt,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
            );
            for j in 0..hidden_size {
                buffers.hidden[j] += buffers.hidden_alt[j];
            }

            // FFN
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
        }

        // Output
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

        kv.advance();

        if i >= 5 {
            times.push(total_start.elapsed());
        }
    }

    let avg: std::time::Duration =
        times.iter().sum::<std::time::Duration>() / times.len() as u32;
    println!("Total: {:?} ({:.1} t/s)", avg, 1.0 / avg.as_secs_f64());
}
