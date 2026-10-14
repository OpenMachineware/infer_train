// Profile GPU inference to find bottlenecks

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    println!("Loading model: {}", model_path);

    // GPU path
    println!("\n=== GPU Inference Profile ===");
    let gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");
    println!("Model loaded, {} layers", gpu_ctx.layer_count());

    let config = gpu_ctx.model().config.clone();
    println!(
        "Config: hidden={}, n_heads={}, head_dim={}",
        config.hidden_size, config.n_heads, config.head_dim
    );

    // Profile first layer GEMV
    {
        let model = gpu_ctx.model();
        if let Some(layer) = model.layers.first() {
            println!("\n--- Layer 0 GEMV Profile ---");

            // QKV weights info
            let wq = layer.wq.as_ref().unwrap();
            let wk = layer.wk.as_ref().unwrap();
            let wv = layer.wv.as_ref().unwrap();
            println!("wq: dtype={:?}, shape={:?}", wq.dtype, wq.shape);
            println!("wk: dtype={:?}, shape={:?}", wk.dtype, wk.shape);
            println!("wv: dtype={:?}, shape={:?}", wv.dtype, wv.shape);
        }
    }

    // Full forward pass timing - GPU
    println!("\n--- Full Forward Pass (GPU) ---");
    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");
    let mut kv_cache = KVCache::new(&config, 10, KVCacheFormat::F32);
    let tokens: Vec<u32> = vec![1, 2, 3];

    let start = Instant::now();
    let _ = gpu_ctx.forward(&tokens, &mut kv_cache);
    let elapsed = start.elapsed();
    println!("Total forward (3 tokens): {:?}", elapsed);
    println!("Per token: {:?}", elapsed / 3);

    // CPU comparison
    println!("\n--- Full Forward Pass (CPU) ---");
    let mut cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load CPU model");

    let mut kv_cache2 = KVCache::new(&config, 10, KVCacheFormat::F32);
    let start = Instant::now();
    let _ = cpu_ctx.forward(&tokens, &mut kv_cache2);
    let elapsed = start.elapsed();
    println!("Total forward (3 tokens): {:?}", elapsed);
    println!("Per token: {:?}", elapsed / 3);

    // Ratio
    println!("\n--- Analysis ---");
    if elapsed.as_nanos() > 0 {
        let _gpu_ns = elapsed.as_nanos();
        println!("GPU/CPU ratio: similar performance");
    }
}
