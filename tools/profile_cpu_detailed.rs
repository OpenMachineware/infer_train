// Detailed CPU profiling - breakdown by operation

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::types::{BlockQ8K, QK_K};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== Detailed CPU Profiling ===\n");
    println!("Model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    let hidden_size = ctx.config.hidden_size;
    println!("Hidden size: {}", hidden_size);

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Warmup
    println!("\n=== Warmup ===");
    let tokens: Vec<u32> = vec![1, 2, 3];
    let _ = ctx.model.forward(&tokens, &mut kv_cache);
    kv_cache.clear();
    println!("Warmup complete");

    // Profile quantization
    println!("\n=== Profile Input Quantization ===");
    let x: Vec<f32> = vec![0.1; hidden_size];
    let nb = (hidden_size + QK_K - 1) / QK_K;
    let mut x_q8k = vec![0u8; nb * std::mem::size_of::<BlockQ8K>()];
    let x_q8k_blocks = unsafe {
        std::slice::from_raw_parts_mut(x_q8k.as_mut_ptr() as *mut BlockQ8K, nb)
    };

    // Measure quantization time
    let n_iter = 100;
    let start = Instant::now();
    for _ in 0..n_iter {
        quantize_row_q8_k_neon(&x, x_q8k_blocks);
    }
    let quant_time = start.elapsed() / n_iter;
    println!(
        "quantize_row_q8_k_neon: {:?} per call ({} iterations)",
        quant_time, n_iter
    );
    println!("  Elements: {}", hidden_size);
    println!(
        "  Throughput: {:.2} GB/s",
        (hidden_size * 4) as f64 / quant_time.as_secs_f64() / 1e9
    );

    // Profile single layer
    println!("\n=== Profile Single Layer ===");
    kv_cache.clear();

    // Create work buffers similar to forward_pooled
    let mut buffers = infer_train::cpu::CpuWorkBuffers::new(
        hidden_size,
        ctx.config.intermediate_size,
        ctx.config.n_heads,
        ctx.config.n_kv_heads,
        ctx.config.head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        ctx.config.vocab_size,
        0,    // qkv_out_size - auto-calculate
        2048, // max_seq_len
        ctx.config.n_layers,
    );

    // Get first layer
    let layer = &ctx.model.layers[0];

    // Manual single token forward with timing
    let token = 1u32;
    ctx.model
        .embedding_lookup_into(token, &mut buffers.hidden)
        .expect("Embedding failed");

    let mut total_norm_time = std::time::Duration::ZERO;
    let mut total_quant_time = std::time::Duration::ZERO;
    let mut total_gemv_time = std::time::Duration::ZERO;
    let mut total_rope_time = std::time::Duration::ZERO;
    let mut total_attn_time = std::time::Duration::ZERO;
    let mut total_ffn_time = std::time::Duration::ZERO;

    // Measure attention norm
    let start = Instant::now();
    let attn_norm_offset = 0 * 2 * hidden_size;
    ctx.model.apply_rms_norm_predequant(
        &buffers.hidden,
        &buffers.norm_weights[attn_norm_offset..attn_norm_offset + hidden_size],
        &mut buffers.hidden_alt,
    );
    total_norm_time += start.elapsed();

    // Measure QKV projection
    if let Some(ref qkv) = layer.attn_qkv {
        let start = Instant::now();
        // The quantization happens inside gemv_into_with_buffers
        // We need to instrument it separately
        let gemv_start = Instant::now();
        ctx.model.gemv_into_with_buffers(
            &buffers.hidden_alt,
            qkv,
            &mut buffers.qkv_buffer,
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );
        let gemv_elapsed = gemv_start.elapsed();
        total_gemv_time += gemv_elapsed;
    }

    // Measure RoPE
    let start = Instant::now();
    // ... (simplified, actual RoPE call would be here)
    total_rope_time += start.elapsed();

    // Measure attention
    let start = Instant::now();
    // ... (simplified, actual attention call would be here)
    total_attn_time += start.elapsed();

    // Measure FFN
    let start = Instant::now();
    // ... (simplified, actual FFN call would be here)
    total_ffn_time += start.elapsed();

    println!("\n  Layer 0 breakdown:");
    println!("    Norm:    {:?}", total_norm_time);
    println!("    GEMV:    {:?}", total_gemv_time);
    println!("    RoPE:    {:?}", total_rope_time);
    println!("    Attn:    {:?}", total_attn_time);
    println!("    FFN:     {:?}", total_ffn_time);

    // Full decode profiling with timing breakdown
    println!("\n=== Full Decode Profiling ===");
    kv_cache.clear();

    let mut decode_tokens = vec![1u32];
    let n_decode = 10;

    let mut total_decode_time = std::time::Duration::ZERO;
    let mut total_quant_all = std::time::Duration::ZERO;
    let mut total_gemv_all = std::time::Duration::ZERO;
    let mut total_other = std::time::Duration::ZERO;

    for i in 0..n_decode {
        let token_start = Instant::now();
        let _ = ctx.model.forward_pooled(
            &decode_tokens,
            &mut kv_cache,
            &mut buffers,
        );
        total_decode_time += token_start.elapsed();

        decode_tokens.clear();
        decode_tokens.push(100 + i as u32);
    }

    println!("\nAverage decode time: {:?}", total_decode_time / n_decode);

    // Estimate operations
    let n_layers = ctx.config.n_layers;
    let gemv_per_token = n_layers * 3 + n_layers * 2; // QKV + FFN (gate + up)
    let quant_per_token = gemv_per_token; // Each GEMV needs input quantization

    println!("\n  Operations per token:");
    println!("    Layers: {}", n_layers);
    println!("    GEMV calls: {}", gemv_per_token);
    println!("    Quant calls: {}", quant_per_token);
    println!(
        "    Estimated quant time: {:?}",
        quant_time * quant_per_token as u32
    );
    println!(
        "    Estimated GEMV time: {:?}",
        total_gemv_time * n_layers as u32
    );

    // Calculate breakdown
    let est_quant = quant_time * quant_per_token as u32;
    let est_gemv = total_gemv_time * n_layers as u32;
    let est_other = total_decode_time - est_quant - est_gemv;

    println!("\n  Time breakdown estimate:");
    println!(
        "    Quantization: {:?} ({:.1}%)",
        est_quant,
        est_quant.as_secs_f64() / total_decode_time.as_secs_f64() * 100.0
    );
    println!(
        "    GEMV:         {:?} ({:.1}%)",
        est_gemv,
        est_gemv.as_secs_f64() / total_decode_time.as_secs_f64() * 100.0
    );
    println!(
        "    Other:        {:?} ({:.1}%)",
        est_other,
        est_other.as_secs_f64() / total_decode_time.as_secs_f64() * 100.0
    );

    // Check thread pool overhead
    println!("\n=== Thread Pool Overhead Check ===");
    let pool = &*infer_train::cpu::thread_pool::GLOBAL_THREAD_POOL;

    // Simple test - measure thread pool execute overhead
    let n_threads = pool.n_threads();
    println!("Thread pool threads: {}", n_threads);

    let mut counter = std::sync::atomic::AtomicUsize::new(0);
    let n_items = 1000;

    let start = Instant::now();
    for _ in 0..n_iter {
        counter.store(0, std::sync::atomic::Ordering::Relaxed);
        pool.execute(
            |counter_ptr, chunk_size, n_items, _user_data| unsafe {
                let counter =
                    &*(counter_ptr as *const std::sync::atomic::AtomicUsize);
                loop {
                    let chunk_start = counter.fetch_add(
                        chunk_size,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    if chunk_start >= n_items {
                        break;
                    }
                }
            },
            n_items,
            64,
            &counter as *const _ as usize,
        );
    }
    let pool_overhead = start.elapsed() / n_iter;
    println!(
        "Thread pool execute (empty work): {:?} per call ({} iterations)",
        pool_overhead, n_iter
    );

    println!("\n=== Summary ===");
    println!("Total decode time: {:?}", total_decode_time / n_decode);
    println!(
        "Thread pool overhead per token: {:?}",
        pool_overhead * (gemv_per_token as u32)
    );
    println!("Quantization time per token: {:?}", est_quant);
    println!();
    println!("Bandwidth analysis:");
    let weights_bytes = 7.0 * 1024.0 * 1024.0 * 1024.0; // ~7GB for 7B model
    let time_per_token = total_decode_time / n_decode;
    let bandwidth = weights_bytes / time_per_token.as_secs_f64();
    println!("  Estimated bandwidth: {:.2} GB/s", bandwidth);
    println!(
        "  Theoretical (100 GB/s): {:.1}% utilization",
        bandwidth / 100.0 * 100.0
    );
}
