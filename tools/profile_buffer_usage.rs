// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Analyze intermediate buffer usage during decode

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::rc::Rc;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        let mut kv_cache = KVCacheGPU::new(
            &device,
            &ctx.config,
            ctx.config.max_position_embeddings,
        );

        // Prefill
        println!("=== Prefilling ===");
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        println!("\n=== Intermediate Buffer Analysis ===");

        // Analyze buffer sizes needed per layer
        let hidden_size = ctx.config.hidden_size;
        let intermediate_size = ctx.config.intermediate_size;
        let n_heads = ctx.config.n_heads;
        let head_dim = ctx.config.head_dim;
        let n_kv_heads = ctx.config.n_kv_heads;

        println!("\n--- Per-Layer Buffer Sizes ---");
        println!("hidden_size: {}", hidden_size);
        println!("intermediate_size: {}", intermediate_size);
        println!(
            "n_heads: {}, n_kv_heads: {}, head_dim: {}",
            n_heads, n_kv_heads, head_dim
        );

        // Calculate buffer sizes
        let norm_out_size = hidden_size * 4; // FP32
        let q_size = n_heads * head_dim * 4;
        let kv_size = n_kv_heads * head_dim * 4;
        let attn_out_size = n_heads * head_dim * 4;
        let ffn_gate_up_size = intermediate_size * 4;
        let ffn_down_size = hidden_size * 4;

        println!("\nBuffer sizes per layer:");
        println!(
            "  norm_out: {} bytes ({:.1} KB)",
            norm_out_size,
            norm_out_size as f64 / 1024.0
        );
        println!("  Q: {} bytes ({:.1} KB)", q_size, q_size as f64 / 1024.0);
        println!("  K: {} bytes ({:.1} KB)", kv_size, kv_size as f64 / 1024.0);
        println!("  V: {} bytes ({:.1} KB)", kv_size, kv_size as f64 / 1024.0);
        println!(
            "  attn_out: {} bytes ({:.1} KB)",
            attn_out_size,
            attn_out_size as f64 / 1024.0
        );
        println!(
            "  ffn_gate_up: {} bytes ({:.1} KB)",
            ffn_gate_up_size,
            ffn_gate_up_size as f64 / 1024.0
        );
        println!(
            "  ffn_down: {} bytes ({:.1} KB)",
            ffn_down_size,
            ffn_down_size as f64 / 1024.0
        );

        // Total intermediate memory per layer
        let per_layer_intermediate = norm_out_size
            + q_size
            + 2 * kv_size
            + attn_out_size
            + 2 * ffn_gate_up_size
            + ffn_down_size;
        println!(
            "\nTotal intermediate per layer: {} bytes ({:.1} KB)",
            per_layer_intermediate,
            per_layer_intermediate as f64 / 1024.0
        );

        let n_layers = ctx.model.layers.len();
        println!("\nFor {} layers:", n_layers);
        println!(
            "  If all layers sequential: {} bytes ({:.1} MB)",
            per_layer_intermediate * n_layers,
            (per_layer_intermediate * n_layers) as f64 / 1024.0 / 1024.0
        );
        println!(
            "  If buffers reused: {} bytes ({:.1} KB)",
            per_layer_intermediate,
            per_layer_intermediate as f64 / 1024.0
        );

        // Check the DAG structure to understand data dependencies
        println!("\n=== DAG Structure Analysis ===");

        let dag = ctx.model.optimized_dag.as_ref().expect("No optimized DAG");

        // Find tensor lifetime - when they're created and when last used
        let mut tensor_first_use: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut tensor_last_use: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for (idx, &node_id) in dag.execution_order.iter().enumerate() {
            let node = &dag.dag.nodes[node_id];

            // Get input and output tensors for this operation
            let (inputs, outputs) = match &node.op {
                infer_train::dag::Operation::RmsNorm {
                    input, output, ..
                } => (vec![*input], vec![*output]),
                infer_train::dag::Operation::Gemv { input, output, .. } => {
                    (vec![*input], vec![*output])
                }
                infer_train::dag::Operation::FusedQKV {
                    input,
                    output_q,
                    output_k,
                    output_v,
                    ..
                } => (vec![*input], vec![*output_q, *output_k, *output_v]),
                infer_train::dag::Operation::RoPE { input, output, .. } => {
                    (vec![*input], vec![*output])
                }
                infer_train::dag::Operation::FlashAttention {
                    output, ..
                } => (vec![], vec![*output]),
                infer_train::dag::Operation::KVCacheUpdate {
                    input, ..
                } => (vec![*input], vec![]),
                infer_train::dag::Operation::FusedSwiGLU {
                    gate,
                    up,
                    output,
                    ..
                } => (vec![*gate, *up], vec![*output]),
                infer_train::dag::Operation::FusedAddNorm {
                    lhs,
                    rhs,
                    output,
                    ..
                } => (vec![*lhs, *rhs], vec![*output]),
                infer_train::dag::Operation::Add { lhs, rhs, output } => {
                    (vec![*lhs, *rhs], vec![*output])
                }
                infer_train::dag::Operation::EmbeddingLookup {
                    output, ..
                } => (vec![], vec![*output]),
                _ => (vec![], vec![]),
            };

            // Track first and last use
            for t in inputs.iter().chain(outputs.iter()) {
                tensor_last_use.insert(*t, idx);
                if !tensor_first_use.contains_key(t) {
                    tensor_first_use.insert(*t, idx);
                }
            }
        }

        // Count concurrent tensors (max buffers needed at any point)
        let mut max_concurrent = 0;
        let mut max_point = 0;

        for i in 0..dag.execution_order.len() {
            let mut concurrent = 0;
            for (&tensor_id, &first) in &tensor_first_use {
                let last = tensor_last_use[&tensor_id];
                if first <= i && i <= last {
                    concurrent += 1;
                }
            }
            if concurrent > max_concurrent {
                max_concurrent = concurrent;
                max_point = i;
            }
        }

        println!("Max concurrent tensors: {}", max_concurrent);
        println!("At execution step: {}", max_point);

        // Estimate total intermediate memory needed
        // This is a lower bound since we don't know exact tensor sizes
        println!(
            "\nEstimated peak intermediate memory: at least {} buffers",
            max_concurrent
        );

        // Compare with llama.cpp's approach
        println!("\n=== Comparison with llama.cpp ===");
        println!("llama.cpp likely:");
        println!("  - Uses in-place operations where possible");
        println!("  - Fuses more operations to reduce intermediates");
        println!("  - Has optimized tensor lifetime management");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
