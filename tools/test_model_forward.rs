// SPDX-License-Identifier: AGPL-3.0-or-later
// Test model forward directly

use infer_train::inference::{Device, InferenceContext, KVCache, KVCacheFormat};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    println!("Testing forward for token 1...");

    // Test standard forward
    kv_cache.clear();
    let result = ctx.model.forward(&[1], &mut kv_cache);

    match result {
        Some(logits) => {
            let sum: f32 = logits.iter().sum();
            let non_zero = logits.iter().filter(|&&x| x != 0.0).count();
            println!("Forward result: len={}, sum={:.4}, non_zero={}, first 5: {:?}",
                logits.len(), sum, non_zero, &logits[..5.min(logits.len())]);
        }
        None => {
            println!("Forward returned None!");
        }
    }

    // Test graph forward
    kv_cache.clear();
    ctx.model.build_optimized_dag();
    let result = ctx.model.forward_graph(1, 0, &mut kv_cache);

    match result {
        Some(logits) => {
            let sum: f32 = logits.iter().sum();
            let non_zero = logits.iter().filter(|&&x| x != 0.0).count();
            println!("Graph forward result: len={}, sum={:.4}, non_zero={}, first 5: {:?}",
                logits.len(), sum, non_zero, &logits[..5.min(logits.len())]);
        }
        None => {
            println!("Graph forward returned None!");
        }
    }
}
