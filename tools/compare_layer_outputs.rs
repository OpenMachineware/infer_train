// SPDX-License-Identifier: AGPL-3.0-or-later
// Compare layer-by-layer outputs between standard and graph dispatch

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

    // Run graph dispatch first to build DAG
    let mut kv_graph = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let _ = ctx.forward_graph(test_token, &mut kv_graph);

    // Check if DAG has FusedSwiGLU
    if let Some(ref dag) = ctx.model.optimized_dag {
        let mut has_fused = false;
        for node in &dag.dag.nodes {
            if matches!(
                node.op,
                infer_train::dag::Operation::FusedSwiGLU { .. }
            ) {
                has_fused = true;
                break;
            }
        }
        println!("DAG has FusedSwiGLU: {}", has_fused);

        // Check FFN structure for layer 0
        println!("\nLayer 0 FFN operations:");
        for node in &dag.dag.nodes {
            if node.layer_idx == Some(0) {
                match &node.op {
                    infer_train::dag::Operation::Gemv {
                        input,
                        weight,
                        output,
                        ..
                    } => {
                        println!(
                            "  Gemv: input={} weight={} output={}",
                            input, weight, output
                        );
                    }
                    infer_train::dag::Operation::SiLU { input, output } => {
                        println!("  SiLU: input={} output={}", input, output);
                    }
                    infer_train::dag::Operation::Mul { lhs, rhs, output } => {
                        println!(
                            "  Mul: lhs={} rhs={} output={}",
                            lhs, rhs, output
                        );
                    }
                    infer_train::dag::Operation::FusedSwiGLU {
                        gate,
                        up,
                        output,
                        ..
                    } => {
                        println!(
                            "  FusedSwiGLU: gate={} up={} output={}",
                            gate, up, output
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    // Compare hidden states after each layer
    println!("\nComparing hidden states (first 5 values):");

    // Standard forward - layer by layer
    let hidden_std = ctx.model.embedding_lookup_impl(test_token).unwrap();
    println!(
        "Embedding std: {:.4} {:.4} {:.4} {:.4} {:.4}",
        hidden_std[0],
        hidden_std[1],
        hidden_std[2],
        hidden_std[3],
        hidden_std[4]
    );

    // Can't easily trace layer-by-layer for standard forward
    // Instead, compare final logits
    let mut kv_std = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits_std = ctx.forward(&[test_token], &mut kv_std).unwrap();

    let mut kv_graph2 = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits_graph = ctx.forward_graph(test_token, &mut kv_graph2).unwrap();

    println!("\nFinal logits comparison:");
    println!(
        "Std (first 10): {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
        logits_std[0],
        logits_std[1],
        logits_std[2],
        logits_std[3],
        logits_std[4],
        logits_std[5],
        logits_std[6],
        logits_std[7],
        logits_std[8],
        logits_std[9]
    );
    println!(
        "Graph (first 10): {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
        logits_graph[0],
        logits_graph[1],
        logits_graph[2],
        logits_graph[3],
        logits_graph[4],
        logits_graph[5],
        logits_graph[6],
        logits_graph[7],
        logits_graph[8],
        logits_graph[9]
    );

    let max_diff = logits_std
        .iter()
        .zip(logits_graph.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\nMax logit difference: {:.4}", max_diff);
}
