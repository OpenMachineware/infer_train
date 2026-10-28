// SPDX-License-Identifier: AGPL-3.0-or-later
// Benchmark for graph-level CPU dispatch

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
    println!("Loading model from {}...", model_path);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    println!("Model config:");
    println!("  Hidden size: {}", ctx.config.hidden_size);
    println!("  Layers: {}", ctx.config.n_layers);
    println!("  Heads: {}", ctx.config.n_heads);
    println!("  KV heads: {}", ctx.config.n_kv_heads);
    println!("  Vocab size: {}", ctx.config.vocab_size);
    println!("  Thread pool size: {}", infer_train::cpu::default_n_threads());

    // Create KV cache
    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Build optimized DAG
    println!("\nBuilding optimized DAG...");
    ctx.model.build_optimized_dag();
    if let Some(ref dag) = ctx.model.optimized_dag {
        println!("  Nodes: {}", dag.node_count());
        println!("  Edges: {}", dag.edge_count());
        println!("  Weight bindings: {}", dag.dag.weight_bindings.len());
    }

    // Test single token
    let test_token = 1u32;

    // Test standard forward
    println!("\n=== Standard forward ===");
    kv_cache.clear();
    let standard_result = ctx.forward(&[test_token], &mut kv_cache);
    match &standard_result {
        Some(result) => {
            println!("  Succeeded, result len: {}", result.len());
            println!(
                "  First 10 logits: {:?}",
                &result[..10.min(result.len())]
            );
            println!(
                "  Non-zero count: {}",
                result.iter().filter(|&&x| x != 0.0).count()
            );
        }
        None => {
            eprintln!("  Failed!");
        }
    }

    // Test graph dispatch
    println!("\n=== Graph dispatch ===");
    kv_cache.clear();
    let graph_result = ctx.forward_graph(test_token, &mut kv_cache);
    match &graph_result {
        Some(result) => {
            println!("  Succeeded, result len: {}", result.len());
            println!(
                "  First 10 logits: {:?}",
                &result[..10.min(result.len())]
            );
            println!(
                "  Non-zero count: {}",
                result.iter().filter(|&&x| x != 0.0).count()
            );
        }
        None => {
            eprintln!("  Failed!");
        }
    }

    // Compare results
    if let (Some(std), Some(graph)) = (&standard_result, &graph_result) {
        println!("\n=== Comparison ===");
        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f32;
        for i in 0..std.len().min(graph.len()) {
            let diff = (std[i] - graph[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            total_diff += diff;
        }
        let avg_diff = total_diff / std.len() as f32;
        println!("  Max difference: {:.6}", max_diff);
        println!("  Avg difference: {:.6}", avg_diff);

        if max_diff < 1e-4 {
            println!("  ✓ Results match!");
        } else {
            println!("  ✗ Results differ!");
        }
    }
}
