use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::ops::sampling::SamplingStrategy;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Inference Benchmark (Zero-Allocation) ===");
    println!("Model: {}", model_path);

    // Load model and create context
    let start = std::time::Instant::now();
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Load time: {:?}", start.elapsed());

    // Check Rayon thread count
    let num_threads = rayon::current_num_threads();
    println!("Rayon threads: {}", num_threads);

    // Use simple token IDs for testing
    let prompt_tokens: Vec<u32> = vec![ctx.config.bos_token_id, 1, 2, 3, 4, 5];
    println!("Prompt tokens: {}", prompt_tokens.len());

    // Create KV cache once
    let mut kv_cache = KVCache::new(
        &ctx.config,
        prompt_tokens.len() + 100,
        KVCacheFormat::F32,
    );

    // Warmup
    println!("\n=== Warmup ===");
    let _ = ctx.forward_pooled(&prompt_tokens, &mut kv_cache);
    println!("Warmup complete");

    // Benchmark decode
    println!("\n=== Decode Benchmark (20 tokens) ===");
    kv_cache.clear();

    // Prefill
    let start = std::time::Instant::now();
    let _ = ctx.forward_pooled(&prompt_tokens, &mut kv_cache);
    let prefill_time = start.elapsed();
    println!("Prefill time: {:?}", prefill_time);

    // Decode
    let mut generated = Vec::with_capacity(20);
    let decode_start = std::time::Instant::now();

    for _ in 0..20 {
        let last_token =
            generated.last().copied().unwrap_or(ctx.config.bos_token_id);

        match ctx.forward_pooled(&[last_token], &mut kv_cache) {
            Some(logits) => {
                // Greedy sampling
                let next_token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap();
                generated.push(next_token);
            }
            None => {
                eprintln!("Warning: forward_pooled returned None");
                break;
            }
        }
    }

    let decode_time = decode_start.elapsed();
    let tps = 20.0 / decode_time.as_secs_f64();

    println!("Tokens: {}", generated.len());
    println!("Time: {:?}", decode_time);
    println!("Speed: {:.2} t/s", tps);

    println!("\n=== Performance Analysis ===");
    let model_size_mb = 387.0;
    let bandwidth_gbps = model_size_mb * tps / 1000.0;
    println!("Effective memory bandwidth: {:.2} GB/s", bandwidth_gbps);

    println!("\nGenerated tokens: {:?}", &generated[..generated.len().min(10)]);
}
