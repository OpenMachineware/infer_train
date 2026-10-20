// Test DAG executor correctness
// Tests DAG building and optimization

use infer_train::dag::DagBuilder;
use infer_train::dag::graph::Operation;
use infer_train::inference::InferenceContext;

fn main() {
    println!("=== DAG Executor Test ===\n");

    // Load model
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/Qwen3-0.6B-Q5_K.gguf".to_string());

    println!("Loading model from: {}", model_path);

    // Load model (use CPU for this test since we're just testing DAG structure)
    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");
    println!("Model loaded: {} layers", ctx.model.layers.len());
    println!("Hidden size: {}", ctx.model.config.hidden_size);
    println!("Heads: {}", ctx.model.config.n_heads);

    // Build DAG
    println!("\n=== Building DAG ===\n");
    let mut builder = DagBuilder::new();
    let dag = builder.build_model_dag(&ctx.model);
    println!("DAG nodes: {}", dag.nodes.len());
    println!("DAG edges: {}", dag.edges.len());
    println!("Weight bindings: {}", dag.weight_bindings.len());
    println!("Entry points: {}", dag.entry_points.len());
    println!("Exit points: {}", dag.exit_points.len());

    // Show sample nodes
    println!("\nSample nodes:");
    for (i, node) in dag.nodes.iter().take(5).enumerate() {
        println!(
            "  Node {}: {:?} (deps: {:?})",
            i,
            std::mem::discriminant(&node.op),
            node.dependencies
        );
    }

    // Optimize DAG
    println!("\n=== Optimizing DAG ===\n");
    let optimized_dag = dag.into_optimized();
    println!("Execution order length: {}", optimized_dag.execution_order.len());

    // Show execution order (first 10)
    println!("\nFirst 10 nodes in execution order:");
    for (i, node_id) in
        optimized_dag.execution_order.iter().take(10).enumerate()
    {
        let node = &optimized_dag.dag.nodes[*node_id];
        println!(
            "  Step {}: Node {} - {:?}",
            i,
            node_id,
            std::mem::discriminant(&node.op)
        );
    }

    // Show node 534 details
    println!("\nNode 534 details:");
    if optimized_dag.dag.nodes.len() > 534 {
        let node = &optimized_dag.dag.nodes[534];
        match &node.op {
            Operation::RmsNorm { input, output, .. } => {
                println!(
                    "  Operation: RmsNorm(input={}, output={})",
                    input, output
                );
                println!("  Dependencies: {:?}", node.dependencies);
            }
            Operation::FusedQKV {
                input, output_q, output_k, output_v, ..
            } => {
                println!(
                    "  Operation: FusedQKV(input={}, outputs={}/{}/{})",
                    input, output_q, output_k, output_v
                );
                println!("  Dependencies: {:?}", node.dependencies);
            }
            Operation::NoOp => {
                println!("  Operation: NoOp");
            }
            _ => {
                println!("  Operation: {:?}", std::mem::discriminant(&node.op));
            }
        }

        // Check what nodes 534's dependencies are
        println!("  Checking dependencies:");
        for &dep_id in &node.dependencies {
            let dep_node = &optimized_dag.dag.nodes[dep_id];
            println!(
                "    Dep {}: {:?} (in exec_order: {})",
                dep_id,
                std::mem::discriminant(&dep_node.op),
                optimized_dag.execution_order.contains(&dep_id)
            );
        }
    } else {
        println!(
            "  DAG has {} nodes, node 534 does not exist",
            optimized_dag.dag.nodes.len()
        );
    }

    // Find which node produces tensor 677
    println!("\nSearching for node that produces tensor 677:");
    for node in &optimized_dag.dag.nodes {
        match &node.op {
            Operation::RmsNorm { output, .. } if *output == 677 => {
                println!("  Node {}: RmsNorm output=677", node.id);
            }
            Operation::Gemv { output, .. } if *output == 677 => {
                println!("  Node {}: Gemv output=677", node.id);
            }
            Operation::Add { output, .. } if *output == 677 => {
                println!("  Node {}: Add output=677", node.id);
            }
            _ => {}
        }
    }

    // Check if node 487 is in execution order
    if let Some(pos) =
        optimized_dag.execution_order.iter().position(|&x| x == 487)
    {
        println!(
            "\nNode 487 (RmsNorm producing 677) is at execution position {}",
            pos
        );
    } else {
        println!("\nNode 487 NOT in execution order!");
    }

    // Check if node 534 is in execution order
    if let Some(pos) =
        optimized_dag.execution_order.iter().position(|&x| x == 534)
    {
        println!("\nNode 534 is at execution position {}", pos);
    } else {
        println!("\nNode 534 NOT in execution order!");
        println!(
            "Execution order length: {}",
            optimized_dag.execution_order.len()
        );
    }

    // Analyze weight bindings
    println!("\n=== Weight Bindings ===\n");
    let mut weight_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (_, tensor) in &optimized_dag.dag.weight_bindings {
        let dtype_name = format!("{:?}", tensor.dtype);
        *weight_counts.entry(dtype_name).or_insert(0) += 1;
    }
    println!("Weight tensor counts by dtype:");
    for (dtype, count) in &weight_counts {
        println!("  {}: {}", dtype, count);
    }

    println!("\n=== Test Complete ===");
    println!("\nSummary:");
    println!("- DAG built with {} nodes", optimized_dag.node_count());
    println!(
        "- {} weight tensors bound",
        optimized_dag.dag.weight_bindings.len()
    );
}
