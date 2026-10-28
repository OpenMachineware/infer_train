// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test DAG building and integration

use infer_train::dag::DagBuilder;
use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    if let (Some(_gpu_ctx), Some(_kv_cache)) =
        (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_ref())
    {
        let model = &ctx.model;
        let layer = &model.layers[0];

        // Build DAG for layer 0
        let mut builder = DagBuilder::new();
        let dag = builder.build_layer_dag(model, layer, 0, 0);

        println!("\n=== DAG Construction ===");
        println!("Nodes: {}", dag.nodes.len());
        println!("Entry points: {:?}", dag.entry_points);
        println!("Exit points: {:?}", dag.exit_points);

        let exec_order = dag.execution_order();
        println!("Execution order: {} nodes", exec_order.len());

        // Print operations
        println!("\n=== Operations ===");
        for (i, node_id) in exec_order.iter().enumerate() {
            let node = &dag.nodes[*node_id];
            println!("{}. Node {} (deps: {:?})", i, node.id, node.dependencies);

            match &node.op {
                infer_train::dag::Operation::RmsNorm { .. } => {
                    println!("   RmsNorm")
                }
                infer_train::dag::Operation::Gemv {
                    out_size,
                    is_transposed,
                    ..
                } => {
                    println!(
                        "   Gemv (out_size: {}, is_transposed: {})",
                        out_size, is_transposed
                    )
                }
                infer_train::dag::Operation::RoPE { n_heads, .. } => {
                    println!("   RoPE (n_heads: {})", n_heads)
                }
                infer_train::dag::Operation::FlashAttention {
                    n_heads,
                    n_kv_heads,
                    ..
                } => {
                    println!(
                        "   FlashAttention (n_heads: {}, n_kv_heads: {})",
                        n_heads, n_kv_heads
                    )
                }
                infer_train::dag::Operation::KVCacheUpdate { .. } => {
                    println!("   KVCacheUpdate")
                }
                infer_train::dag::Operation::SiLU { .. } => println!("   SiLU"),
                infer_train::dag::Operation::Mul { .. } => println!("   Mul"),
                infer_train::dag::Operation::Add { .. } => println!("   Add"),
                _ => println!("   Other operation"),
            }
        }

        // Compare with llama.cpp's graph size
        println!("\n=== Comparison ===");
        println!("Our DAG: {} nodes per layer", dag.nodes.len());
        println!("llama.cpp: ~34 nodes per layer (959 nodes / 28 layers)");
        println!(
            "Note: llama.cpp includes embedding lookup, output projection, etc."
        );
    }
}
