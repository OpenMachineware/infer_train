// SPDX-License-Identifier: AGPL-3.0-or-later
// Trace first few operations to find divergence

use infer_train::dag::graph::Operation;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use infer_train::model::LlamaModel;

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

    // Test token
    let test_token = 1u32;

    // 1. Embedding lookup (standard)
    println!("\n=== Step 1: Embedding ===");
    let embed =
        ctx.model.embedding_lookup_impl(test_token).expect("Embedding failed");
    println!("Standard embedding:");
    println!("  Len: {}", embed.len());
    println!("  First 5: {:?}", &embed[..5.min(embed.len())]);
    println!("  Sum: {}", embed.iter().sum::<f32>());

    // 2. Layer 0 attention norm (standard)
    println!("\n=== Step 2: RMS Norm ===");
    let layer = &ctx.model.layers[0];
    let attn_norm = layer.attention_norm.as_ref().expect("No attention_norm");
    let mut normed = vec![0.0f32; embed.len()];
    ctx.model.apply_rms_norm_into(&embed, attn_norm, &mut normed);
    println!("Standard normed:");
    println!("  Len: {}", normed.len());
    println!("  First 5: {:?}", &normed[..5.min(normed.len())]);
    println!("  Sum: {}", normed.iter().sum::<f32>());

    // 3. Q projection (standard)
    println!("\n=== Step 3: Q projection ===");
    let layer = &ctx.model.layers[0];
    println!("Layer 0 has q_norm: {}", layer.q_norm.is_some());
    println!("Layer 0 has k_norm: {}", layer.k_norm.is_some());
    let wq = layer.wq.clone().expect("No wq");
    let wq_dtype = wq.dtype;
    let wq_shape = wq.shape.clone();
    let wq_bytes_len = wq.bytes().len();
    let q_standard = ctx
        .model
        .gemv_impl(&normed, &wq, wq.shape[1])
        .expect("Q projection failed");
    println!("Standard Q:");
    println!("  Len: {}", q_standard.len());
    println!("  First 5: {:?}", &q_standard[..5.min(q_standard.len())]);
    println!("  Sum: {}", q_standard.iter().sum::<f32>());
    println!("  wq dtype: {:?}", wq_dtype);
    println!("  wq shape: {:?}", wq_shape);
    println!("  wq bytes len: {}", wq_bytes_len);

    // Now test graph execution step by step
    println!("\n=== Graph execution ===");

    // Build optimized DAG
    ctx.model.build_optimized_dag();
    let dag = ctx.model.optimized_dag.as_ref().expect("DAG not built");

    println!("DAG nodes: {}", dag.node_count());
    println!("Weight bindings: {}", dag.dag.weight_bindings.len());

    // Check maximum tensor ID
    let mut max_tensor_id = 0;
    for node_id in dag.execution_order.iter() {
        let node = &dag.dag.nodes[*node_id];
        match &node.op {
            Operation::EmbeddingLookup { output, .. } => {
                max_tensor_id = max_tensor_id.max(*output);
            }
            Operation::RmsNorm { output, .. } => {
                max_tensor_id = max_tensor_id.max(*output);
            }
            Operation::Gemv { output, .. } => {
                max_tensor_id = max_tensor_id.max(*output);
            }
            Operation::RoPE { output, .. } => {
                max_tensor_id = max_tensor_id.max(*output);
            }
            Operation::FlashAttention { output, .. } => {
                max_tensor_id = max_tensor_id.max(*output);
            }
            Operation::Add { output, .. } => {
                max_tensor_id = max_tensor_id.max(*output);
            }
            Operation::FusedSwiGLU { output, .. } => {
                max_tensor_id = max_tensor_id.max(*output);
            }
            _ => {}
        }
    }
    println!("Max tensor ID: {}", max_tensor_id);

    // Check if weight_id=5 is correctly bound
    println!("\nChecking weight binding for weight_id=5 (should be Q weight):");
    if let Some(dag_wq) = dag.dag.weight_bindings.get(&5) {
        println!("  DAG weight dtype: {:?}", dag_wq.dtype);
        println!("  DAG weight shape: {:?}", dag_wq.shape);
        println!("  DAG weight bytes len: {}", dag_wq.bytes().len());

        // Try GEMV with DAG weight
        let dag_result = ctx.model.gemv_impl(&normed, dag_wq, dag_wq.shape[1]);
        match dag_result {
            Some(r) => {
                println!("  DAG gemv result sum: {}", r.iter().sum::<f32>());
            }
            None => {
                println!("  DAG gemv returned None!");
            }
        }
    } else {
        println!("  Weight not found!");
    }

    // Look at the first few operations in execution order
    println!("\nFirst 15 operations:");
    for (i, node_id) in dag.execution_order.iter().take(15).enumerate() {
        let node = &dag.dag.nodes[*node_id];
        print!("  [{}] node_id={} ", i, node_id);
        match &node.op {
            Operation::EmbeddingLookup { output, .. } => {
                println!("EmbeddingLookup: output={}", output);
            }
            Operation::RmsNorm { input, output, weight, .. } => {
                println!(
                    "RmsNorm: input={}, weight={}, output={}",
                    input, weight, output
                );
                if let Some(w) = dag.dag.weight_bindings.get(weight) {
                    println!(
                        "       -> weight dtype={:?}, shape={:?}",
                        w.dtype, w.shape
                    );
                }
            }
            Operation::Gemv { input, output, weight, .. } => {
                println!(
                    "Gemv: input={}, weight={}, output={}",
                    input, weight, output
                );
                if let Some(w) = dag.dag.weight_bindings.get(weight) {
                    println!(
                        "       -> weight dtype={:?}, shape={:?}",
                        w.dtype, w.shape
                    );
                }
            }
            Operation::RoPE { input, output, n_heads, .. } => {
                println!(
                    "RoPE: input={}, output={}, n_heads={}",
                    input, output, n_heads
                );
            }
            Operation::KVCacheUpdate { input, layer_idx, is_key, .. } => {
                println!(
                    "KVCacheUpdate: input={}, layer={}, is_key={}",
                    input, layer_idx, is_key
                );
            }
            Operation::FlashAttention { q, output, layer_idx, .. } => {
                println!(
                    "FlashAttention: q={}, output={}, layer={}",
                    q, output, layer_idx
                );
            }
            Operation::Add { lhs, rhs, output } => {
                println!("Add: lhs={}, rhs={}, output={}", lhs, rhs, output);
            }
            Operation::FusedSwiGLU { gate, up, output, .. } => {
                println!(
                    "FusedSwiGLU: gate={}, up={}, output={}",
                    gate, up, output
                );
            }
            _ => {
                println!("Other operation");
            }
        }
    }

    // Execute manually with debug
    println!("\n=== Manual graph execution ===");
    let weights = &dag.dag.weight_bindings;
    let mut tensors = vec![vec![], vec![]]; // Initialize with some capacity

    // Execute first 3 operations manually
    for (i, node_id) in dag.execution_order.iter().take(8).enumerate() {
        let node = &dag.dag.nodes[*node_id];
        match &node.op {
            Operation::EmbeddingLookup { output, .. } => {
                let embed = ctx
                    .model
                    .embedding_lookup_impl(test_token)
                    .expect("Embedding failed");
                if *output >= tensors.len() {
                    tensors.resize(*output + 1, vec![]);
                }
                tensors[*output] = embed;
                println!(
                    "[{}] EmbeddingLookup: output[{}] sum={}",
                    i,
                    output,
                    tensors[*output].iter().sum::<f32>()
                );
            }
            Operation::RmsNorm { input, weight, output, .. } => {
                let input_data = tensors[*input].clone();
                let w = weights.get(weight).expect("Weight not found");
                if *output >= tensors.len() {
                    tensors.resize(*output + 1, vec![]);
                }
                tensors[*output].resize(input_data.len(), 0.0);
                ctx.model.apply_rms_norm_into(
                    &input_data,
                    w,
                    &mut tensors[*output],
                );
                let sum = tensors[*output].iter().sum::<f32>();
                println!(
                    "[{}] RmsNorm: input[{}] sum={}, output[{}] sum={}",
                    i,
                    input,
                    input_data.iter().sum::<f32>(),
                    output,
                    sum
                );
            }
            Operation::Gemv { input, weight, output, out_size } => {
                let input_data = tensors[*input].clone();
                let w = weights.get(weight).expect("Weight not found");
                println!(
                    "[{}] Gemv: weight_id={}, weight_dtype={:?}, weight_shape={:?}, out_size={}",
                    i, weight, w.dtype, w.shape, out_size
                );
                println!(
                    "       input_data len={}, first 3 values: {:?}",
                    input_data.len(),
                    &input_data[..3.min(input_data.len())]
                );
                if *output >= tensors.len() {
                    tensors.resize(*output + 1, vec![]);
                }
                // Use out_size from DAG, not weight shape
                let result = ctx.model.gemv_impl(&input_data, w, *out_size);
                match result {
                    Some(r) => {
                        let sum = r.iter().sum::<f32>();
                        let non_zero = r.iter().filter(|&&x| x != 0.0).count();
                        tensors[*output] = r;
                        println!(
                            "[{}] Gemv: input[{}] sum={}, output[{}] sum={}, len={}, non_zero={}",
                            i,
                            input,
                            input_data.iter().sum::<f32>(),
                            output,
                            sum,
                            tensors[*output].len(),
                            non_zero
                        );
                        if sum == 0.0 && non_zero == 0 {
                            println!(
                                "       WARNING: All zeros! Weight bytes len: {}",
                                w.bytes().len()
                            );
                        }
                    }
                    None => {
                        println!(
                            "[{}] Gemv FAILED! gemv_impl returned None",
                            i
                        );
                    }
                }
            }
            _ => {}
        }
    }
}
