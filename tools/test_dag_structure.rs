// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test DAG structure for final norm and output projection

fn main() {
    // Create a minimal test model
    use infer_train::model::{LlamaConfig, LlamaModel, Tensor};
    use infer_train::dag::builder::DagBuilder;

    let config = LlamaConfig {
        hidden_size: 256,
        intermediate_size: 512,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 1000,
        norm_eps: 1e-5,
        head_dim: 64,
        rope_freq_base: 10000.0,
    };

    // Create minimal model with weights
    let mut model = LlamaModel::new(config.clone());

    // Add embedding weight
    let embed_weight = Tensor {
        data: vec![0.0f32; config.vocab_size * config.hidden_size],
        shape: vec![config.vocab_size, config.hidden_size],
        ..Default::default()
    };
    model.tok_embeddings = Some(embed_weight);

    // Add output weight
    let output_weight = Tensor {
        data: vec![0.0f32; config.vocab_size * config.hidden_size],
        shape: vec![config.vocab_size, config.hidden_size],
        ..Default::default()
    };
    model.output = Some(output_weight);

    // Add final norm weight
    let norm_weight = Tensor {
        data: vec![1.0f32; config.hidden_size],
        shape: vec![config.hidden_size],
        ..Default::default()
    };
    model.norm = Some(norm_weight);

    // Build DAG
    println!("Building DAG for 1-layer model...");
    let dag = DagBuilder::new().build_model_dag(&model);

    println!("\nDAG structure:");
    println!("  Nodes: {}", dag.nodes.len());
    println!("  Entry points: {:?}", dag.entry_points);
    println!("  Exit points: {:?}", dag.exit_points);

    // Check final nodes
    if let Some(&exit_node_id) = dag.exit_points.last() {
        let exit_node = &dag.nodes[exit_node_id];
        println!("\nExit node:");
        println!("  ID: {}", exit_node_id);
        println!("  Op: {:?}", std::mem::discriminant(&exit_node.op));

        // Check if it's the vocab projection
        if let infer_train::dag::graph::Operation::Gemv { input, output, out_size, .. } = &exit_node.op {
            println!("  Input tensor: {}", input);
            println!("  Output tensor: {}", output);
            println!("  Output size (vocab): {}", out_size);

            // Check if input comes from norm
            // The norm should have been fused with this Gemv
            if let infer_train::dag::graph::Operation::FusedNormLinear { input, output, out_size, .. } = &exit_node.op {
                println!("\n✓ Final norm fused with vocab projection!");
            } else {
                println!("\n✗ Final norm not fused with vocab projection");
            }
        }
    }

    // Count operations
    let mut op_counts = std::collections::HashMap::new();
    for node in &dag.nodes {
        let key = format!("{:?}", std::mem::discriminant(&node.op));
        *op_counts.entry(key).or_insert(0) += 1;
    }

    println!("\nOperation counts:");
    for (op, count) in op_counts {
        println!("  {}: {}", op, count);
    }
}
