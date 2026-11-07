// Exact copy of measure_decode.rs logic
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

    println!("=== Copy of measure_decode.rs ===");

    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    let mut buffer_pool = CpuWorkBuffersPool::new(&ctx.model);
    buffer_pool.preallocate(1);
    let mut buffers = buffer_pool.allocate().unwrap();

    // Prefill with single token (using context's forward_pooled!)
    let _ = ctx.forward_pooled(&[1], &mut kv);

    // Measure decode (using buffer pool's buffers)
    let mut total_times = Vec::new();

    for i in 0..20 {
        let total_start = Instant::now();

        // Embedding lookup
        ctx.model.embedding_lookup_into(100, &mut buffers.hidden);

        // Layers
        for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
            ctx.model.forward_layer_decode_single_thread(
                layer,
                layer_idx,
                &mut kv,
                &mut buffers,
            );
        }
        kv.advance();

        // Output projection
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

        total_times.push(total_start.elapsed());
    }

    let avg_total: std::time::Duration =
        total_times[5..].iter().sum::<std::time::Duration>()
            / (total_times.len() - 5) as u32;

    println!("Total decode: {:?}", avg_total);
    println!("Speed: {:.1} t/s", 1.0 / avg_total.as_secs_f64());
}
