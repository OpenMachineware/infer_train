// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Count kernel dispatches per decode step

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::dag::{DagNode, Operation, OptimizedDag};
        use infer_train::inference::{Device, InferenceContext};
        use infer_train::model::LlamaModel;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");

        println!("=== Decode Operation Count ===");

        // Get the optimized DAG
        let optimized_dag =
            ctx.model.optimized_dag.as_ref().expect("No optimized DAG");
        let dag = &optimized_dag.dag;

        println!("Total nodes: {}", dag.nodes.len());
        println!(
            "Execution order length: {}",
            optimized_dag.execution_order.len()
        );

        // Count operations by type
        let mut gemv_count = 0;
        let mut norm_count = 0;
        let mut rope_count = 0;
        let mut flash_attn_count = 0;
        let mut kv_update_count = 0;
        let mut silu_count = 0;
        let mut mul_count = 0;
        let mut add_count = 0;
        let mut fused_qkv_count = 0;
        let mut fused_swiglu_count = 0;
        let mut embedding_count = 0;
        let mut other_count = 0;

        for &node_idx in &optimized_dag.execution_order {
            let node = &dag.nodes[node_idx];
            match &node.op {
                Operation::Gemv { .. } => gemv_count += 1,
                Operation::RmsNorm { .. } => norm_count += 1,
                Operation::RoPE { .. } => rope_count += 1,
                Operation::FlashAttention { .. } => flash_attn_count += 1,
                Operation::KVCacheUpdate { .. } => kv_update_count += 1,
                Operation::SiLU { .. } => silu_count += 1,
                Operation::Mul { .. } => mul_count += 1,
                Operation::Add { .. } => add_count += 1,
                Operation::FusedQKV { .. } => fused_qkv_count += 1,
                Operation::FusedSwiGLU { .. } => fused_swiglu_count += 1,
                Operation::EmbeddingLookup { .. } => embedding_count += 1,
                _ => {
                    other_count += 1;
                }
            }
        }

        println!("\nOperation counts:");
        println!("  GEMV: {}", gemv_count);
        println!("  RMS Norm: {}", norm_count);
        println!("  RoPE: {}", rope_count);
        println!("  Flash Attention: {}", flash_attn_count);
        println!("  KV Cache Update: {}", kv_update_count);
        println!("  SiLU: {}", silu_count);
        println!("  Mul: {}", mul_count);
        println!("  Add: {}", add_count);
        println!("  FusedQKV: {}", fused_qkv_count);
        println!("  FusedSwiGLU: {}", fused_swiglu_count);
        println!("  Embedding: {}", embedding_count);
        println!("  Other: {}", other_count);

        // Estimate actual dispatches
        // FusedQKV = 3 dispatches (fallback) or 1 dispatch (fused)
        // FusedSwiGLU = 1 dispatch (already fused)
        let estimated_dispatches = gemv_count
            + norm_count
            + rope_count
            + flash_attn_count
            + kv_update_count
            + silu_count
            + mul_count
            + add_count
            + fused_qkv_count * 3
            + fused_swiglu_count
            + embedding_count;
        println!(
            "\nEstimated dispatches (if FusedQKV fallback): {}",
            estimated_dispatches
        );
        println!(
            "Estimated dispatches (if FusedQKV fused): {}",
            estimated_dispatches - fused_qkv_count * 2
        );

        // Calculate expected
        let n_layers = ctx.model.layers.len();
        println!("\nExpected (per layer):");
        println!("  Attention Norm: 1");
        println!("  Q projection: 1 GEMV");
        println!("  K projection: 1 GEMV");
        println!("  V projection: 1 GEMV");
        println!("  RoPE for Q: 1");
        println!("  RoPE for K: 1");
        println!("  KV update K: 1");
        println!("  KV update V: 1");
        println!("  Flash Attention: 1");
        println!("  Output projection: 1 GEMV");
        println!("  FFN Norm: 1");
        println!("  Gate projection: 1 GEMV");
        println!("  Up projection: 1 GEMV");
        println!("  SiLU: 1");
        println!("  Mul: 1");
        println!("  Down projection: 1 GEMV");
        println!("  Add residual: 1");
        println!("  Total: 17 operations per layer");

        let expected = n_layers * 17;
        println!("\nExpected total for {} layers: {}", n_layers, expected);
        println!("Actual total: {}", optimized_dag.execution_order.len());

        // Show fused operations
        println!("\n=== Fused Operations ===");
        for &node_idx in &optimized_dag.execution_order {
            let node = &dag.nodes[node_idx];
            if matches!(
                &node.op,
                Operation::FusedGateUp { .. }
                    | Operation::FusedSwiGLU { .. }
                    | Operation::FusedAddNorm { .. }
                    | Operation::FusedNormLinear { .. }
            ) {
                println!("  {:?}", node.op);
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
