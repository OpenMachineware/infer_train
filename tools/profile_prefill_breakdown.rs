// Profile prefill timing breakdown
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

    println!("=== Prefill Breakdown ===");
    println!("Tokens: 64");

    // Measure just the forward_pooled call
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    let tokens: Vec<u32> = (1..=64).collect();
    let start = Instant::now();
    let _ = ctx.forward_pooled(&tokens, &mut kv);
    let total = start.elapsed();

    println!("\nTotal prefill: {:?}", total);
    println!("Speed: {:.1} t/s", 64.0 / total.as_secs_f64());
}
