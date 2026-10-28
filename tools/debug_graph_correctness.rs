// SPDX-License-Identifier: AGPL-3.0-or-later
// Debug tool to compare standard forward vs graph dispatch step by step

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
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

    // Test single token
    let test_token = 1u32;
    println!("\nTesting with token: {}", test_token);

    // Compare embeddings first
    println!("\n=== Embedding comparison ===");
    let embed_std = ctx.model.embedding_lookup_impl(test_token).unwrap();
    println!("  Standard embedding (first 10): {:?}", &embed_std[..10]);

    // Build DAG
    ctx.model.build_optimized_dag();
    if let Some(ref dag) = ctx.model.optimized_dag {
        println!("\n=== DAG structure ===");
        println!("  Nodes: {}", dag.node_count());
        println!("  Edges: {}", dag.edge_count());

        // Print first 20 nodes to see FFN operations
        for (i, node) in dag.dag.nodes.iter().take(20).enumerate() {
            println!(
                "  Node {}: {:?} (layer: {:?})",
                i, node.op, node.layer_idx
            );
        }
    }

    // Test standard forward
    println!("\n=== Standard forward ===");
    let mut kv_std = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits_std = ctx.forward(&[test_token], &mut kv_std).unwrap();
    println!(
        "  Final logits (first 10): {:?}",
        &logits_std[..10.min(logits_std.len())]
    );

    // Test graph dispatch
    println!("\n=== Graph dispatch ===");
    let mut kv_graph = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits_graph = ctx.forward_graph(test_token, &mut kv_graph).unwrap();
    println!(
        "  Final logits (first 10): {:?}",
        &logits_graph[..10.min(logits_graph.len())]
    );

    // Compare
    println!("\n=== Comparison ===");
    let graph = logits_graph;
    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;
    let mut diff_count = 0;

    for i in 0..logits_std.len().min(graph.len()) {
        let diff = (logits_std[i] - graph[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.001 {
            total_diff += diff;
            diff_count += 1;
        }
    }

    let avg_diff =
        if diff_count > 0 { total_diff / diff_count as f32 } else { 0.0 };

    println!("  Max difference: {:.6}", max_diff);
    println!("  Avg difference: {:.6}", avg_diff);
    println!("  Differing values: {}", diff_count);
}
