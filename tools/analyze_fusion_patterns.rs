// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Analyze fusion patterns in the DAG

use infer_train::dag::Operation;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};

        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");
        let model = &ctx.model;

        // Check if DAG exists
        let optimized_dag =
            model.optimized_dag.as_ref().expect("DAG not built");
        let dag = &optimized_dag.dag;

        println!("=== Fusion Pattern Analysis ===\n");

        // Find GEMV + ADD patterns
        let mut gemv_add_count = 0;
        let mut gemv_add_positions = Vec::new();

        for i in 0..optimized_dag.execution_order.len().saturating_sub(1) {
            let node_idx = optimized_dag.execution_order[i];
            let next_node_idx = optimized_dag.execution_order[i + 1];

            let node = &dag.nodes[node_idx];
            let next_node = &dag.nodes[next_node_idx];

            if matches!(node.op, Operation::Gemv { .. })
                && matches!(next_node.op, Operation::Add { .. })
            {
                gemv_add_count += 1;
                gemv_add_positions.push((node_idx, next_node_idx));
            }
        }

        println!("GEMV + ADD patterns: {}", gemv_add_count);
        if gemv_add_count > 0 {
            println!("  First few positions:");
            for (i, (gemv_idx, add_idx)) in
                gemv_add_positions.iter().take(5).enumerate()
            {
                let gemv_node = &dag.nodes[*gemv_idx];
                println!(
                    "    {}: GEMV -> Add (layer {})",
                    i + 1,
                    gemv_node.layer_idx.unwrap_or(999)
                );
            }
        }

        // Find RMS_NORM + MUL + ADD patterns
        let mut norm_mul_add_count = 0;
        let mut norm_mul_add_positions = Vec::new();

        for i in 0..optimized_dag.execution_order.len().saturating_sub(2) {
            let node_idx = optimized_dag.execution_order[i];
            let next_node_idx = optimized_dag.execution_order[i + 1];
            let next_next_node_idx = optimized_dag.execution_order[i + 2];

            let node = &dag.nodes[node_idx];
            let next_node = &dag.nodes[next_node_idx];
            let next_next_node = &dag.nodes[next_next_node_idx];

            if matches!(node.op, Operation::RmsNorm { .. })
                && matches!(next_node.op, Operation::Mul { .. })
                && matches!(next_next_node.op, Operation::Add { .. })
            {
                norm_mul_add_count += 1;
                norm_mul_add_positions.push((
                    node_idx,
                    next_node_idx,
                    next_next_node_idx,
                ));
            }
        }

        println!("\nRMS_NORM + MUL + ADD patterns: {}", norm_mul_add_count);
        if norm_mul_add_count > 0 {
            println!("  First few positions:");
            for (i, (norm_idx, _mul_idx, _add_idx)) in
                norm_mul_add_positions.iter().take(5).enumerate()
            {
                let norm_node = &dag.nodes[*norm_idx];
                println!(
                    "    {}: RmsNorm -> Mul -> Add (layer {})",
                    i + 1,
                    norm_node.layer_idx.unwrap_or(999)
                );
            }
        }

        // Count current fused operations
        let mut fused_qkv_count = 0;
        let mut fused_swiglu_count = 0;
        let mut fused_add_norm_count = 0;

        for &node_idx in &optimized_dag.execution_order {
            let node = &dag.nodes[node_idx];
            match &node.op {
                Operation::FusedQKV { .. } => fused_qkv_count += 1,
                Operation::FusedSwiGLU { .. } => fused_swiglu_count += 1,
                Operation::FusedAddNorm { .. } => fused_add_norm_count += 1,
                _ => {}
            }
        }

        println!("\n=== Current Fused Operations ===");
        println!("FusedQKV: {}", fused_qkv_count);
        println!("FusedSwiGLU: {}", fused_swiglu_count);
        println!("FusedAddNorm: {}", fused_add_norm_count);

        // Estimate potential savings
        println!("\n=== Potential Savings from Fusion ===");
        let dispatch_overhead_us = 3.6; // From profiling
        let total_nodes = optimized_dag.execution_order.len();

        println!("Total DAG nodes: {}", total_nodes);

        let gemv_add_savings_us = gemv_add_count as f64 * dispatch_overhead_us;
        println!("\nGEMV + ADD fusion:");
        println!("  Dispatches saved: {}", gemv_add_count);
        println!("  Time saved: {:.1} µs", gemv_add_savings_us);

        let norm_mul_add_savings_us =
            norm_mul_add_count as f64 * 2.0 * dispatch_overhead_us; // 2 dispatches saved
        println!("\nRMS_NORM + MUL + ADD fusion:");
        println!(
            "  Dispatches saved: {} (2 per pattern)",
            norm_mul_add_count * 2
        );
        println!("  Time saved: {:.1} µs", norm_mul_add_savings_us);

        let total_savings_us = gemv_add_savings_us + norm_mul_add_savings_us;
        let current_decode_us = 8100.0; // From benchmark

        println!("\n=== Total Potential Improvement ===");
        println!(
            "Total dispatches saved: {}",
            gemv_add_count + norm_mul_add_count * 2
        );
        println!("Total time saved: {:.1} µs", total_savings_us);
        println!(
            "Percentage improvement: {:.1}%",
            total_savings_us / current_decode_us * 100.0
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
