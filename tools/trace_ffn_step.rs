// SPDX-License-Identifier: AGPL-3.0-or-later
// Detailed step-by-step trace of FFN operations

use infer_train::dag::Operation;
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

    // Get layer 0 weights
    let layer = &ctx.model.layers[0];

    // Check FFN weights
    println!("Layer 0 FFN weights:");
    if let Some(ref w1) = layer.w1 {
        println!("  w1 (gate): shape={:?} dtype={:?}", w1.shape, w1.dtype);
    }
    if let Some(ref w2) = layer.w2 {
        println!("  w2 (down): shape={:?} dtype={:?}", w2.shape, w2.dtype);
    }
    if let Some(ref w3) = layer.w3 {
        println!("  w3 (up): shape={:?} dtype={:?}", w3.shape, w3.dtype);
    }

    // Manual FFN computation
    let test_token = 1u32;
    let hidden = ctx.model.embedding_lookup_impl(test_token).unwrap();

    // Layer 0: Attention norm -> QKV -> Attention -> Residual -> FFN norm -> FFN
    // Skip attention, just test FFN with hidden after embedding
    let hidden_size = ctx.config.hidden_size;

    // FFN norm (layer 0)
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let normed =
        ctx.model.apply_rms_norm_impl(&hidden, ffn_norm, hidden_size).unwrap();

    println!(
        "\nAfter FFN norm (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        normed[0], normed[1], normed[2], normed[3], normed[4]
    );

    // Gate projection
    let w1 = layer.w1.as_ref().unwrap();
    let intermediate_size = w1.shape[1];
    let gate = ctx.model.gemv_impl(&normed, w1, intermediate_size).unwrap();

    // Up projection
    let w3 = layer.w3.as_ref().unwrap();
    let up = ctx.model.gemv_impl(&normed, w3, intermediate_size).unwrap();

    println!(
        "Gate (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        gate[0], gate[1], gate[2], gate[3], gate[4]
    );
    println!(
        "Up (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        up[0], up[1], up[2], up[3], up[4]
    );

    // SwiGLU: gate * silu(up)
    let mut hidden_ffn = vec![0.0f32; intermediate_size];
    for i in 0..intermediate_size {
        let silu = up[i] / (1.0 + (-up[i]).exp());
        hidden_ffn[i] = gate[i] * silu;
    }

    println!(
        "SwiGLU (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        hidden_ffn[0],
        hidden_ffn[1],
        hidden_ffn[2],
        hidden_ffn[3],
        hidden_ffn[4]
    );

    // Down projection
    let w2 = layer.w2.as_ref().unwrap();
    let ffn_out = ctx.model.gemv_impl(&hidden_ffn, w2, hidden_size).unwrap();

    println!(
        "FFN out (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        ffn_out[0], ffn_out[1], ffn_out[2], ffn_out[3], ffn_out[4]
    );

    // Now check what the graph dispatch does
    println!("\n--- Graph dispatch FFN ---");

    // Build DAG and trace tensor 16 (FFN norm input)
    ctx.model.build_optimized_dag();
    if let Some(ref dag) = ctx.model.optimized_dag {
        // Find the FFN norm node for layer 0
        for (idx, node) in dag.dag.nodes.iter().enumerate() {
            if node.layer_idx == Some(0) {
                match &node.op {
                    Operation::RmsNorm { input, weight, output, .. } => {
                        if *weight == 17 {
                            // FFN norm weight
                            println!(
                                "Node {}: RmsNorm input={} output={}",
                                idx, input, output
                            );
                        }
                    }
                    Operation::Gemv { input, weight, output, .. } => {
                        if *weight == 19 || *weight == 21 || *weight == 25 {
                            // Gate, Up, Down
                            println!(
                                "Node {}: Gemv input={} weight={} output={}",
                                idx, input, weight, output
                            );
                        }
                    }
                    Operation::SiLU { input, output } => {
                        println!(
                            "Node {}: SiLU input={} output={}",
                            idx, input, output
                        );
                    }
                    Operation::Mul { lhs, rhs, output } => {
                        println!(
                            "Node {}: Mul lhs={} rhs={} output={}",
                            idx, lhs, rhs, output
                        );
                    }
                    _ => {}
                }
            }
        }
    }
}
