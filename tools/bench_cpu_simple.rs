// Simple CPU inference benchmark

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Inference Benchmark ===\n");
    println!("Model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Model loaded successfully");

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Warmup
    println!("\n=== Warmup (6 tokens) ===");
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let _ = ctx.model.forward(&tokens, &mut kv_cache);
    kv_cache.clear();
    println!("Warmup complete");

    // Benchmark decode
    println!("\n=== Decode Benchmark (20 tokens) ===");
    let mut decode_tokens = vec![1u32];

    let start = std::time::Instant::now();
    for i in 0..20 {
        match ctx.model.forward(&decode_tokens, &mut kv_cache) {
            Some(_logits) => {}
            None => {
                eprintln!("Forward failed at token {}", i);
                break;
            }
        }
        decode_tokens.clear();
        decode_tokens.push(100 + (kv_cache.seq_len() as u32) % 100);
    }
    let elapsed = start.elapsed();

    println!("Tokens: 20");
    println!("Time: {:?}", elapsed);
    println!("Speed: {:.2} t/s", 20.0 / elapsed.as_secs_f64());
}
