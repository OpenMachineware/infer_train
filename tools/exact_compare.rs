use infer_train::cpu::CpuWorkBuffersPool;
use infer_train::inference::{
    InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use std::time::{Duration, Instant};

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
    let n_layers = ctx.model.layers.len();

    println!(
        "hidden_size={}, vocab_size={}, n_layers={}",
        hidden_size, vocab_size, n_layers
    );

    // EXACT COPY of measure_decode.rs logic
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let mut buffer_pool = CpuWorkBuffersPool::new(&ctx.model);
    buffer_pool.preallocate(1);
    let mut buffers = buffer_pool.allocate().unwrap();

    let _ = ctx.forward_pooled(&[1], &mut kv);

    let mut times1 = Vec::new();
    for i in 0..20 {
        let total_start = Instant::now();
        ctx.model.embedding_lookup_into(100, &mut buffers.hidden);
        for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
            ctx.model.forward_layer_decode_single_thread(
                layer,
                layer_idx,
                &mut kv,
                &mut buffers,
            );
        }
        kv.advance();
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
        if i >= 5 {
            times1.push(total_start.elapsed());
        }
    }

    let avg1: Duration = times1.iter().sum::<Duration>() / times1.len() as u32;
    println!(
        "Method 1 (exact measure_decode): {:?} ({:.1} t/s)",
        avg1,
        1.0 / avg1.as_secs_f64()
    );

    // Method 2: forward_pooled
    let mut kv2 = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let _ = ctx.forward_pooled(&[1], &mut kv2);

    let mut times2 = Vec::new();
    for i in 0..20 {
        let start = Instant::now();
        let _ = ctx.forward_pooled(&[100], &mut kv2);
        if i >= 5 {
            times2.push(start.elapsed());
        }
    }

    let avg2: Duration = times2.iter().sum::<Duration>() / times2.len() as u32;
    println!(
        "Method 2 (forward_pooled): {:?} ({:.1} t/s)",
        avg2,
        1.0 / avg2.as_secs_f64()
    );

    println!("Ratio: {:.2}x", avg1.as_secs_f64() / avg2.as_secs_f64());
}
