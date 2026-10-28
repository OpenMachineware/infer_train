// SPDX-License-Identifier: AGPL-3.0-or-later
// Check if standard and graph dispatch produce same final token

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
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let test_token = 1u32;

    // Standard forward
    let mut kv_std = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits_std = ctx.forward(&[test_token], &mut kv_std).unwrap();
    let max_idx_std = logits_std
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // Graph dispatch
    let mut kv_graph = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits_graph = ctx.forward_graph(test_token, &mut kv_graph).unwrap();
    let max_idx_graph = logits_graph
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!(
        "Standard top token: {} (logit: {:.4})",
        max_idx_std, logits_std[max_idx_std]
    );
    println!(
        "Graph top token: {} (logit: {:.4})",
        max_idx_graph, logits_graph[max_idx_graph]
    );

    if max_idx_std == max_idx_graph {
        println!("✅ SAME TOKEN - Error is acceptable");
    } else {
        println!("❌ DIFFERENT TOKEN - Error is too large");
        // Check top 5 tokens
        let mut std_ranked: Vec<_> = logits_std.iter().enumerate().collect();
        std_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let mut graph_ranked: Vec<_> =
            logits_graph.iter().enumerate().collect();
        graph_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        println!("\nTop 5 standard: {:?}", &std_ranked[..5]);
        println!("Top 5 graph: {:?}", &graph_ranked[..5]);
    }
}
