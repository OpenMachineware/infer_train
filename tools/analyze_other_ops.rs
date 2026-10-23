// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Analyze all "Other" operations in detail

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::dag::Operation;
        use infer_train::inference::{Device, InferenceContext};

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");

        println!("=== Analyzing All Operations ===");

        let optimized_dag =
            ctx.model.optimized_dag.as_ref().expect("No optimized DAG");
        let dag = &optimized_dag.dag;

        // Count all operation types
        let mut op_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for &node_idx in &optimized_dag.execution_order {
            let node = &dag.nodes[node_idx];
            let op_name = match &node.op {
                Operation::EmbeddingLookup { .. } => "EmbeddingLookup",
                Operation::RmsNorm { .. } => "RmsNorm",
                Operation::Gemv { .. } => "Gemv",
                Operation::FusedQKV { .. } => "FusedQKV",
                Operation::FusedQKVWeight { .. } => "FusedQKVWeight",
                Operation::FusedGateUp { .. } => "FusedGateUp",
                Operation::RoPE { .. } => "RoPE",
                Operation::FlashAttention { .. } => "FlashAttention",
                Operation::KVCacheUpdate { .. } => "KVCacheUpdate",
                Operation::SiLU { .. } => "SiLU",
                Operation::Mul { .. } => "Mul",
                Operation::Add { .. } => "Add",
                Operation::FusedSwiGLU { .. } => "FusedSwiGLU",
                Operation::FusedNormLinear { .. } => "FusedNormLinear",
                Operation::FusedAddNorm { .. } => "FusedAddNorm",
                Operation::FusedGemvAdd { .. } => "FusedGemvAdd",
                Operation::NoOp => "NoOp",
                Operation::ExpertRouting { .. } => "ExpertRouting",
                Operation::ExpertCompute { .. } => "ExpertCompute",
            };
            *op_counts.entry(op_name.to_string()).or_insert(0) += 1;
        }

        println!("\nOperation breakdown:");
        let mut total = 0;
        for (op_name, count) in op_counts.iter() {
            println!("  {}: {}", op_name, count);
            total += count;
        }
        println!("\nTotal: {}", total);

        // Calculate kernel dispatches
        // Each operation that has a Metal kernel implementation is a dispatch
        let mut dispatch_count = 0;
        for (op_name, count) in op_counts.iter() {
            // Operations that DO NOT dispatch kernels:
            // - NoOp: does nothing
            // - Exit: just marks output
            let is_dispatch = !matches!(op_name.as_str(), "NoOp" | "Exit");
            if is_dispatch {
                dispatch_count += count;
            }
        }

        println!("\nKernel dispatches: {}", dispatch_count);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
