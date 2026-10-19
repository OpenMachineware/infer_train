// Test DAG executor correctness
// Tests DAG building and optimization

use infer_train::dag::{DagBuilder, DagOptimizer};
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

    // Check for fusion opportunities
    println!("\n=== Fusion Analysis ===\n");
    let fusion_patterns =
        DagOptimizer::find_fusion_patterns(&optimized_dag.dag);
    println!("Fusion patterns found: {}", fusion_patterns.len());
    for pattern in &fusion_patterns {
        println!("  - {}", pattern.name);
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
    println!("- {} fusion opportunities identified", fusion_patterns.len());
    println!(
        "- {} weight tensors bound",
        optimized_dag.dag.weight_bindings.len()
    );
}
