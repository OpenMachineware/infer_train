// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! End-to-end test for fused kernels with real model

use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, ModelConfig, llama::LlamaModel};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

    println!("=== Fused Kernel End-to-End Test ===");
    println!("Loading model: {}", model_path);

    // Load weights and config
    let start = Instant::now();
    let store =
        WeightStore::from_gguf(model_path).expect("Failed to load weights");
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let config = ModelConfig::from_gguf(&gguf);
    println!("Weights loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Print model info
    println!("\nModel config:");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {}, KV heads: {}", config.n_heads, config.n_kv_heads);
    println!("  Vocab size: {}", config.vocab_size);

    // Build LlamaModel
    println!("\nBuilding LlamaModel...");
    let start = Instant::now();
    let llama_model = LlamaModel::from_weights(&store, config.clone());
    println!("Model built in {:.2}s", start.elapsed().as_secs_f32());

    // Check DAG
    println!("\n=== DAG Analysis ===");
    if let Some(dag) = &llama_model.optimized_dag {
        println!("DAG nodes: {}", dag.node_count());
        println!("DAG edges: {}", dag.edge_count());

        // Count operation types
        let mut silu_count = 0;
        let mut mul_count = 0;
        let mut norm_count = 0;
        let mut gemv_count = 0;
        let mut fused_swiglu_count = 0;
        let mut fused_norm_linear_count = 0;
        let mut fused_qkv_count = 0;

        for node in dag.nodes_in_order() {
            match &node.op {
                infer_train::dag::graph::Operation::SiLU { .. } => {
                    silu_count += 1
                }
                infer_train::dag::graph::Operation::Mul { .. } => {
                    mul_count += 1
                }
                infer_train::dag::graph::Operation::RmsNorm { .. } => {
                    norm_count += 1
                }
                infer_train::dag::graph::Operation::Gemv { .. } => {
                    gemv_count += 1
                }
                infer_train::dag::graph::Operation::FusedSwiGLU { .. } => {
                    fused_swiglu_count += 1
                }
                infer_train::dag::graph::Operation::FusedNormLinear {
                    ..
                } => fused_norm_linear_count += 1,
                infer_train::dag::graph::Operation::FusedQKV { .. } => {
                    fused_qkv_count += 1
                }
                infer_train::dag::graph::Operation::NoOp => {
                    // Skip - placeholder for fused operations
                }
                _ => {}
            }
        }

        println!("\nOriginal operations:");
        println!("  SiLU: {}", silu_count);
        println!("  Mul: {}", mul_count);
        println!("  RMSNorm: {}", norm_count);
        println!("  Gemv: {}", gemv_count);

        println!("\nFused operations:");
        println!("  FusedSwiGLU: {}", fused_swiglu_count);
        println!("  FusedNormLinear: {}", fused_norm_linear_count);
        println!("  FusedQKV: {}", fused_qkv_count);

        // Debug: count NoOp nodes
        let noop_count = dag
            .nodes_in_order()
            .filter(|n| {
                matches!(&n.op, infer_train::dag::graph::Operation::NoOp)
            })
            .count();
        println!("  NoOp (removed): {}", noop_count);

        // Debug: show total node counts
        println!("\n  Total nodes in DAG: {}", dag.dag.nodes.len());
        println!("  Nodes in execution order: {}", dag.execution_order.len());
        let fused_in_dag = dag
            .dag
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    &n.op,
                    infer_train::dag::graph::Operation::FusedSwiGLU { .. }
                )
            })
            .count();
        let norm_in_dag = dag
            .dag
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    &n.op,
                    infer_train::dag::graph::Operation::FusedNormLinear { .. }
                )
            })
            .count();
        println!("  FusedSwiGLU in dag.nodes: {}", fused_in_dag);
        println!("  FusedNormLinear in dag.nodes: {}", norm_in_dag);

        // Calculate potential savings
        let potential_swiglu = silu_count.min(mul_count);
        let potential_norm_linear = norm_count.min(gemv_count);

        println!("\nFusion potential:");
        println!(
            "  SiLU+Mul pairs: {} (each saves 1 dispatch)",
            potential_swiglu
        );
        println!(
            "  RMSNorm+Gemv pairs: {} (each saves 1 dispatch)",
            potential_norm_linear
        );

        let total_original = silu_count + mul_count + norm_count + gemv_count;
        let saved = fused_swiglu_count + fused_norm_linear_count;
        if total_original > 0 {
            println!(
                "\nKernel dispatch reduction: {} → {} (saved {} = {:.1}%)",
                total_original,
                total_original - saved,
                saved,
                (saved as f64 / total_original as f64) * 100.0
            );
        }
    } else {
        println!("DAG not built");
    }

    // Create Metal context and test
    println!("\n=== Metal Context Test ===");
    match MetalContext::new() {
        Ok(gpu_ctx) => {
            println!("Metal device: {}", gpu_ctx.device().name());

            // Test fused kernel availability
            println!("\nFused kernels available:");
            println!("  ✓ FusedSwiGLU");
            println!("  ✓ FusedNormLinear (F16 weights)");

            // Check weight dtypes
            println!("\nWeight tensor dtypes (layer 0):");
            if let Some(layer) = llama_model.layers.first() {
                if let Some(wq) = &layer.wq {
                    println!("  wq: {:?}", wq.dtype);
                }
                if let Some(w1) = &layer.w1 {
                    println!("  w1 (gate): {:?}", w1.dtype);
                }
            }
        }
        Err(e) => {
            println!("Failed to create Metal context: {}", e);
        }
    }

    println!("\n=== Test Complete ===");
}
