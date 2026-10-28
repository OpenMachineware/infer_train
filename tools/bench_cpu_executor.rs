// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Benchmark for CPU Executor (zero-allocation version)

use infer_train::inference::cpu_executor::CpuExecutor;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::llama::LlamaModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Executor Benchmark ===\n");
    println!("Model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Load time: {:?}", std::time::Instant::now());

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Create CPU executor
    let mut executor = CpuExecutor::new(&ctx.model);
    println!(
        "CPU Executor memory usage: {} KB",
        executor.memory_usage() / 1024
    );

    // Warmup
    println!("\n=== Warmup ===");
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let _ = executor.forward(&ctx.model, &tokens, &mut kv_cache);
    kv_cache.clear();

    // Benchmark decode
    println!("\n=== Decode Benchmark (20 tokens) ===");
    let mut decode_tokens = vec![1u32];

    let start = std::time::Instant::now();
    for _ in 0..20 {
        let _logits = executor
            .forward(&ctx.model, &decode_tokens, &mut kv_cache)
            .expect("Forward failed");
        decode_tokens.clear();
        decode_tokens.push(100 + (kv_cache.seq_len() as u32) % 100);
    }
    let elapsed = start.elapsed();

    println!("Tokens: 20");
    println!("Time: {:?}", elapsed);
    println!("Speed: {:.2} t/s", 20.0 / elapsed.as_secs_f64());

    // Compare with regular forward
    println!("\n=== Comparison with Regular Forward ===");
    kv_cache.clear();
    let mut decode_tokens = vec![1u32];

    let start = std::time::Instant::now();
    for _ in 0..20 {
        let _logits = ctx
            .model
            .forward(&decode_tokens, &mut kv_cache)
            .expect("Forward failed");
        decode_tokens.clear();
        decode_tokens.push(100 + (kv_cache.seq_len() as u32) % 100);
    }
    let elapsed_regular = start.elapsed();

    println!(
        "Regular forward: {:.2} t/s",
        20.0 / elapsed_regular.as_secs_f64()
    );
    println!(
        "Executor speedup: {:.2}x",
        elapsed_regular.as_secs_f64() / elapsed.as_secs_f64()
    );
}
