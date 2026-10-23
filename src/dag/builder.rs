// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG Builder - Model architecture → DAG
//!
//! Transforms model architecture into a computation graph (DAG)

use super::graph::{ComputeDag, DagNode, DevicePreference, Operation};
use crate::model::{LlamaLayer, LlamaModel};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::backend::Backend;

pub struct DagBuilder {
    dag: ComputeDag,
    next_tensor_id: usize,
}

impl DagBuilder {
    pub fn new() -> Self {
        Self { dag: ComputeDag::new(), next_tensor_id: 0 }
    }

    /// Build DAG for a single layer forward pass
    pub fn build_layer_dag(
        &mut self,
        model: &LlamaModel,
        layer: &LlamaLayer,
        layer_idx: usize,
        input_tensor_id: usize,
    ) -> ComputeDag {
        // Reset
        self.dag = ComputeDag::new();
        self.next_tensor_id = input_tensor_id + 1;

        let hidden_size = model.config.hidden_size;
        let head_dim = model.config.head_dim;

        // 1. Attention RMS Norm
        let normed_id = self.alloc_tensor();
        let norm_weight_id = self.alloc_tensor(); // Will be bound to weight later
        let norm_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RmsNorm {
                input: input_tensor_id,
                weight: norm_weight_id,
                output: normed_id,
                epsilon: 1e-5,
            },
            dependencies: vec![],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let norm_node_id = self.dag.add_node(norm_node);

        // 2. Q/K/V Projections
        let q_out_size =
            layer.wq.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);
        let k_out_size =
            layer.wk.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);
        let v_out_size =
            layer.wv.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);

        let q_id = self.alloc_tensor();
        let q_weight_id = self.alloc_tensor();
        let q_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: normed_id,
                weight: q_weight_id,
                output: q_id,
                out_size: q_out_size,
            },
            dependencies: vec![norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let q_node_id = self.dag.add_node(q_node);

        let k_id = self.alloc_tensor();
        let k_weight_id = self.alloc_tensor();
        let k_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: normed_id,
                weight: k_weight_id,
                output: k_id,
                out_size: k_out_size,
            },
            dependencies: vec![norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let k_node_id = self.dag.add_node(k_node);

        let v_id = self.alloc_tensor();
        let v_weight_id = self.alloc_tensor();
        let v_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: normed_id,
                weight: v_weight_id,
                output: v_id,
                out_size: v_out_size,
            },
            dependencies: vec![norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let v_node_id = self.dag.add_node(v_node);

        // 3. RoPE Q/K
        let n_heads = q_out_size / head_dim;
        let n_kv_heads = k_out_size / head_dim;

        let q_rope_id = self.alloc_tensor();
        let q_rope_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RoPE {
                input: q_id,
                output: q_rope_id,
                pos: 0, // Will be set during execution
                n_heads,
                head_dim,
            },
            dependencies: vec![q_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let q_rope_node_id = self.dag.add_node(q_rope_node);

        let k_rope_id = self.alloc_tensor();
        let k_rope_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RoPE {
                input: k_id,
                output: k_rope_id,
                pos: 0,
                n_heads: n_kv_heads,
                head_dim,
            },
            dependencies: vec![k_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let k_rope_node_id = self.dag.add_node(k_rope_node);

        // 4. KV Cache Updates
        let k_cache_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::KVCacheUpdate {
                input: k_rope_id,
                layer_idx,
                is_key: true, // K cache update
            },
            dependencies: vec![k_rope_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let k_cache_node_id = self.dag.add_node(k_cache_node);

        let v_cache_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::KVCacheUpdate {
                input: v_id,
                layer_idx,
                is_key: false, // V cache update
            },
            dependencies: vec![v_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let v_cache_node_id = self.dag.add_node(v_cache_node);

        // 5. Flash Attention
        let attn_out_id = self.alloc_tensor();
        let attn_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::FlashAttention {
                q: q_rope_id,
                output: attn_out_id,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            },
            dependencies: vec![
                q_rope_node_id,
                k_cache_node_id,
                v_cache_node_id,
            ],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let attn_node_id = self.dag.add_node(attn_node);

        // 6. Output Projection
        let wo_out_id = self.alloc_tensor();
        let wo_weight_id = self.alloc_tensor();
        let wo_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: attn_out_id,
                weight: wo_weight_id,
                output: wo_out_id,
                out_size: hidden_size,
            },
            dependencies: vec![attn_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let wo_node_id = self.dag.add_node(wo_node);

        // 7. Residual Add
        let residual_id = self.alloc_tensor();
        let residual_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Add {
                lhs: input_tensor_id,
                rhs: wo_out_id,
                output: residual_id,
            },
            dependencies: vec![wo_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let residual_node_id = self.dag.add_node(residual_node);

        // 8. FFN Norm
        let ffn_normed_id = self.alloc_tensor();
        let ffn_norm_weight_id = self.alloc_tensor();
        let ffn_norm_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RmsNorm {
                input: residual_id,
                weight: ffn_norm_weight_id,
                output: ffn_normed_id,
                epsilon: 1e-5,
            },
            dependencies: vec![residual_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let ffn_norm_node_id = self.dag.add_node(ffn_norm_node);

        // 9. Gate/Up Projections (parallel)
        let intermediate_size =
            layer.w1.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size * 4);

        let gate_id = self.alloc_tensor();
        let gate_weight_id = self.alloc_tensor();
        let gate_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: ffn_normed_id,
                weight: gate_weight_id,
                output: gate_id,
                out_size: intermediate_size,
            },
            dependencies: vec![ffn_norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let gate_node_id = self.dag.add_node(gate_node);

        let up_id = self.alloc_tensor();
        let up_weight_id = self.alloc_tensor();
        let up_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: ffn_normed_id,
                weight: up_weight_id,
                output: up_id,
                out_size: intermediate_size,
            },
            dependencies: vec![ffn_norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let up_node_id = self.dag.add_node(up_node);

        // 10. SiLU activation
        let silu_id = self.alloc_tensor();
        let silu_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::SiLU { input: gate_id, output: silu_id },
            dependencies: vec![gate_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let silu_node_id = self.dag.add_node(silu_node);

        // 11. Element-wise mul
        let ffn_hidden_id = self.alloc_tensor();
        let mul_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Mul {
                lhs: silu_id,
                rhs: up_id,
                output: ffn_hidden_id,
            },
            dependencies: vec![silu_node_id, up_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let mul_node_id = self.dag.add_node(mul_node);

        // 12. Down projection
        let ffn_out_id = self.alloc_tensor();
        let down_weight_id = self.alloc_tensor();
        if let Some(weight) = &layer.w2 {
            self.dag.bind_weight(down_weight_id, weight.clone());
        }
        let down_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: ffn_hidden_id,
                weight: down_weight_id,
                output: ffn_out_id,
                out_size: hidden_size,
            },
            dependencies: vec![mul_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let down_node_id = self.dag.add_node(down_node);

        // 13. Final residual add
        let output_id = self.alloc_tensor();
        let final_residual_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Add {
                lhs: residual_id,
                rhs: ffn_out_id,
                output: output_id,
            },
            dependencies: vec![residual_node_id, down_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None,
        };
        let final_residual_node_id = self.dag.add_node(final_residual_node);

        // Mark output
        self.dag.exit_points.push(final_residual_node_id);

        // Add edges
        self.add_all_edges();

        std::mem::take(&mut self.dag)
    }

    /// Build DAG for entire model forward pass
    ///
    /// This constructs a complete DAG including:
    /// - Embedding lookup (entry point)
    /// - All transformer layers
    /// - Final normalization
    /// - Output projection
    pub fn build_model_dag(&mut self, model: &LlamaModel) -> ComputeDag {
        // Reset
        self.dag = ComputeDag::new();
        self.next_tensor_id = 0;

        let hidden_size = model.config.hidden_size;

        // 1. Embedding lookup (entry point)
        let embed_tensor_id = self.alloc_tensor();
        let embedding_weight_id = self.alloc_tensor();
        if let Some(weight) = &model.tok_embeddings {
            self.dag.bind_weight(embedding_weight_id, weight.clone());
        }
        let embed_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::EmbeddingLookup {
                token_id: 0, // Will be set at execution time
                embedding_weight: embedding_weight_id,
                output: embed_tensor_id,
            },
            dependencies: vec![],
            device_preference: DevicePreference::Cpu, // Embedding on CPU
            layer_idx: None,                          // No layer for embedding
        };
        let embed_node_id = self.dag.add_node(embed_node);

        // Process each layer
        let mut layer_input = embed_tensor_id;
        let mut last_layer_node_id = embed_node_id;
        for layer_idx in 0..model.layers.len() {
            let layer = &model.layers[layer_idx];
            let (layer_output, last_node) = self.build_layer_dag_inline(
                model,
                layer,
                layer_idx,
                layer_input,
                last_layer_node_id,
            );
            layer_input = layer_output;
            last_layer_node_id = last_node;
        }

        // 2. Final norm
        let final_norm_id = self.alloc_tensor();
        let norm_weight_id = self.alloc_tensor();
        if let Some(weight) = &model.norm {
            self.dag.bind_weight(norm_weight_id, weight.clone());
        }
        let final_norm_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RmsNorm {
                input: layer_input,
                weight: norm_weight_id,
                output: final_norm_id,
                epsilon: 1e-5,
            },
            dependencies: vec![self.dag.nodes.len() - 1], // Depends on last layer
            device_preference: DevicePreference::Gpu,
            layer_idx: None, // Final norm is not part of any layer
        };
        let final_norm_node_id = self.dag.add_node(final_norm_node);

        // 3. Output projection
        let output_id = self.alloc_tensor();
        let output_weight_id = self.alloc_tensor();
        if let Some(weight) = &model.output {
            self.dag.bind_weight(output_weight_id, weight.clone());
        }
        let vocab_size =
            model.output.as_ref().map(|o| o.shape[0]).unwrap_or(hidden_size);
        let output_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: final_norm_id,
                weight: output_weight_id,
                output: output_id,
                out_size: vocab_size,
            },
            dependencies: vec![final_norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: None, // Output projection is not part of any layer
        };
        let output_node_id = self.dag.add_node(output_node);

        // Mark output
        self.dag.exit_points.push(output_node_id);

        // Add edges
        self.add_all_edges();

        std::mem::take(&mut self.dag)
    }

    /// Build DAG for a single layer, inline into existing DAG
    /// Returns (output_tensor_id, last_node_id)
    fn build_layer_dag_inline(
        &mut self,
        model: &LlamaModel,
        layer: &LlamaLayer,
        layer_idx: usize,
        input_tensor_id: usize,
        last_node_id: usize,
    ) -> (usize, usize) {
        let hidden_size = model.config.hidden_size;
        let head_dim = model.config.head_dim;

        // 1. Attention RMS Norm
        let normed_id = self.alloc_tensor();
        let norm_weight_id = self.alloc_tensor();
        if let Some(weight) = &layer.attention_norm {
            self.dag.bind_weight(norm_weight_id, weight.clone());
        }
        let norm_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RmsNorm {
                input: input_tensor_id,
                weight: norm_weight_id,
                output: normed_id,
                epsilon: 1e-5,
            },
            dependencies: vec![last_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let norm_node_id = self.dag.add_node(norm_node);

        // 2. Q/K/V Projections
        // Determine output sizes and create nodes based on model architecture
        let (
            q_id,
            k_id,
            v_id,
            q_node_id,
            k_node_id,
            v_node_id,
            q_out_size,
            k_out_size,
            v_out_size,
        ) = if let Some(ref qkv_weight) = layer.attn_qkv {
            // Fused QKV: single weight tensor for Q/K/V (Qwen3.8-style)
            let q_out_size = model.config.n_heads * model.config.head_dim;
            let k_out_size = model.config.n_kv_heads * model.config.head_dim;
            let v_out_size = model.config.n_kv_heads * model.config.head_dim;

            let q_id = self.alloc_tensor();
            let k_id = self.alloc_tensor();
            let v_id = self.alloc_tensor();
            let qkv_weight_id = self.alloc_tensor();
            self.dag.bind_weight(qkv_weight_id, qkv_weight.clone());

            if layer_idx == 0 && std::env::var("INFER_DEBUG_DAG").is_ok() {
                eprintln!(
                    "[DAG BUILD] Layer 0: fused QKV weight_id={}",
                    qkv_weight_id
                );
            }

            let qkv_node = DagNode {
                id: self.dag.nodes.len(),
                op: Operation::FusedQKVWeight {
                    input: normed_id,
                    qkv_weight: qkv_weight_id,
                    output_q: q_id,
                    output_k: k_id,
                    output_v: v_id,
                    q_out_size,
                    k_out_size,
                    v_out_size,
                    layer_idx,
                },
                dependencies: vec![norm_node_id],
                device_preference: DevicePreference::Gpu,
                layer_idx: Some(layer_idx),
            };
            let qkv_node_id = self.dag.add_node(qkv_node);

            if layer_idx == 0 && std::env::var("INFER_DEBUG_DAG").is_ok() {
                eprintln!(
                    "[DAG BUILD] Layer 0: FusedQKVWeight created q_id={}, k_id={}, v_id={}",
                    q_id, k_id, v_id
                );
            }

            // For fused QKV, all Q/K/V depend on the same node
            (
                q_id,
                k_id,
                v_id,
                qkv_node_id,
                qkv_node_id,
                qkv_node_id,
                q_out_size,
                k_out_size,
                v_out_size,
            )
        } else {
            // Separate Q/K/V weights
            let q_out_size =
                layer.wq.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);
            let k_out_size =
                layer.wk.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);
            let v_out_size =
                layer.wv.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size);

            let q_id = self.alloc_tensor();
            let q_weight_id = self.alloc_tensor();
            if let Some(weight) = &layer.wq {
                self.dag.bind_weight(q_weight_id, weight.clone());
            }
            if layer_idx == 0 && std::env::var("INFER_DEBUG_DAG").is_ok() {
                eprintln!(
                    "[DAG BUILD] Layer 0: q_weight_id={}, weight bound={}",
                    q_weight_id,
                    layer.wq.is_some()
                );
            }
            let q_node = DagNode {
                id: self.dag.nodes.len(),
                op: Operation::Gemv {
                    input: normed_id,
                    weight: q_weight_id,
                    output: q_id,
                    out_size: q_out_size,
                },
                dependencies: vec![norm_node_id],
                device_preference: DevicePreference::Gpu,
                layer_idx: Some(layer_idx),
            };
            let q_node_id = self.dag.add_node(q_node);

            let k_id = self.alloc_tensor();
            let k_weight_id = self.alloc_tensor();
            if let Some(weight) = &layer.wk {
                self.dag.bind_weight(k_weight_id, weight.clone());
            }
            if layer_idx == 0 && std::env::var("INFER_DEBUG_DAG").is_ok() {
                eprintln!(
                    "[DAG BUILD] Layer 0: k_weight_id={}, weight bound={}",
                    k_weight_id,
                    layer.wk.is_some()
                );
            }
            let k_node = DagNode {
                id: self.dag.nodes.len(),
                op: Operation::Gemv {
                    input: normed_id,
                    weight: k_weight_id,
                    output: k_id,
                    out_size: k_out_size,
                },
                dependencies: vec![norm_node_id],
                device_preference: DevicePreference::Gpu,
                layer_idx: Some(layer_idx),
            };
            let k_node_id = self.dag.add_node(k_node);

            let v_id = self.alloc_tensor();
            let v_weight_id = self.alloc_tensor();
            if let Some(weight) = &layer.wv {
                self.dag.bind_weight(v_weight_id, weight.clone());
            }
            if layer_idx == 0 && std::env::var("INFER_DEBUG_DAG").is_ok() {
                eprintln!(
                    "[DAG BUILD] Layer 0: v_weight_id={}, weight bound={}",
                    v_weight_id,
                    layer.wv.is_some()
                );
            }
            let v_node = DagNode {
                id: self.dag.nodes.len(),
                op: Operation::Gemv {
                    input: normed_id,
                    weight: v_weight_id,
                    output: v_id,
                    out_size: v_out_size,
                },
                dependencies: vec![norm_node_id],
                device_preference: DevicePreference::Gpu,
                layer_idx: Some(layer_idx),
            };
            let v_node_id = self.dag.add_node(v_node);

            (
                q_id, k_id, v_id, q_node_id, k_node_id, v_node_id, q_out_size,
                k_out_size, v_out_size,
            )
        };

        // 3. RoPE Q/K
        let n_heads = q_out_size / head_dim;
        let n_kv_heads = k_out_size / head_dim;

        let q_rope_id = self.alloc_tensor();
        let q_rope_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RoPE {
                input: q_id,
                output: q_rope_id,
                pos: 0,
                n_heads,
                head_dim,
            },
            dependencies: vec![q_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let q_rope_node_id = self.dag.add_node(q_rope_node);

        let k_rope_id = self.alloc_tensor();
        let k_rope_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RoPE {
                input: k_id,
                output: k_rope_id,
                pos: 0,
                n_heads: n_kv_heads,
                head_dim,
            },
            dependencies: vec![k_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let k_rope_node_id = self.dag.add_node(k_rope_node);

        // 4. KV Cache Updates
        let k_cache_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::KVCacheUpdate {
                input: k_rope_id,
                layer_idx,
                is_key: true, // K cache update
            },
            dependencies: vec![k_rope_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let k_cache_node_id = self.dag.add_node(k_cache_node);

        let v_cache_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::KVCacheUpdate {
                input: v_id,
                layer_idx,
                is_key: false, // V cache update
            },
            dependencies: vec![v_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let v_cache_node_id = self.dag.add_node(v_cache_node);

        // 5. Flash Attention
        let attn_out_id = self.alloc_tensor();
        let attn_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::FlashAttention {
                q: q_rope_id,
                output: attn_out_id,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            },
            dependencies: vec![
                q_rope_node_id,
                k_cache_node_id,
                v_cache_node_id,
            ],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let attn_node_id = self.dag.add_node(attn_node);

        // 6. Output Projection
        let wo_out_id = self.alloc_tensor();
        let wo_weight_id = self.alloc_tensor();
        if let Some(weight) = &layer.wo {
            self.dag.bind_weight(wo_weight_id, weight.clone());
        }
        let wo_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: attn_out_id,
                weight: wo_weight_id,
                output: wo_out_id,
                out_size: hidden_size,
            },
            dependencies: vec![attn_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let wo_node_id = self.dag.add_node(wo_node);

        // 7. Residual Add
        let residual_id = self.alloc_tensor();
        let residual_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Add {
                lhs: input_tensor_id,
                rhs: wo_out_id,
                output: residual_id,
            },
            dependencies: vec![wo_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let residual_node_id = self.dag.add_node(residual_node);

        // 8. FFN Norm
        let ffn_normed_id = self.alloc_tensor();
        let ffn_norm_weight_id = self.alloc_tensor();
        if let Some(weight) = &layer.ffn_norm {
            self.dag.bind_weight(ffn_norm_weight_id, weight.clone());
        }
        let ffn_norm_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::RmsNorm {
                input: residual_id,
                weight: ffn_norm_weight_id,
                output: ffn_normed_id,
                epsilon: 1e-5,
            },
            dependencies: vec![residual_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let ffn_norm_node_id = self.dag.add_node(ffn_norm_node);

        // 9. Gate/Up Projections (parallel)
        let intermediate_size =
            layer.w1.as_ref().map(|w| w.shape[1]).unwrap_or(hidden_size * 4);

        let gate_id = self.alloc_tensor();
        let gate_weight_id = self.alloc_tensor();
        if let Some(weight) = &layer.w1 {
            self.dag.bind_weight(gate_weight_id, weight.clone());
        }
        let gate_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: ffn_normed_id,
                weight: gate_weight_id,
                output: gate_id,
                out_size: intermediate_size,
            },
            dependencies: vec![ffn_norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let gate_node_id = self.dag.add_node(gate_node);

        let up_id = self.alloc_tensor();
        let up_weight_id = self.alloc_tensor();
        if let Some(weight) = &layer.w3 {
            self.dag.bind_weight(up_weight_id, weight.clone());
        }
        let up_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: ffn_normed_id,
                weight: up_weight_id,
                output: up_id,
                out_size: intermediate_size,
            },
            dependencies: vec![ffn_norm_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let up_node_id = self.dag.add_node(up_node);

        // 10. SiLU activation
        let silu_id = self.alloc_tensor();
        let silu_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::SiLU { input: gate_id, output: silu_id },
            dependencies: vec![gate_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let silu_node_id = self.dag.add_node(silu_node);

        // 11. Element-wise mul
        let ffn_hidden_id = self.alloc_tensor();
        let mul_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Mul {
                lhs: silu_id,
                rhs: up_id,
                output: ffn_hidden_id,
            },
            dependencies: vec![silu_node_id, up_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let mul_node_id = self.dag.add_node(mul_node);

        // 12. Down projection
        let ffn_out_id = self.alloc_tensor();
        let down_weight_id = self.alloc_tensor();
        if let Some(weight) = &layer.w2 {
            self.dag.bind_weight(down_weight_id, weight.clone());
        }
        let down_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Gemv {
                input: ffn_hidden_id,
                weight: down_weight_id,
                output: ffn_out_id,
                out_size: hidden_size,
            },
            dependencies: vec![mul_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let down_node_id = self.dag.add_node(down_node);

        // 13. Final residual add
        let output_id = self.alloc_tensor();
        let final_residual_node = DagNode {
            id: self.dag.nodes.len(),
            op: Operation::Add {
                lhs: residual_id,
                rhs: ffn_out_id,
                output: output_id,
            },
            dependencies: vec![residual_node_id, down_node_id],
            device_preference: DevicePreference::Gpu,
            layer_idx: Some(layer_idx),
        };
        let final_residual_node_id = self.dag.add_node(final_residual_node);

        (output_id, final_residual_node_id)
    }

    fn alloc_tensor(&mut self) -> usize {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        id
    }

    fn add_all_edges(&mut self) {
        // This will be called to add edges based on operations
        // For now, dependencies are tracked in node.dependencies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LlamaConfig, LlamaLayer, LlamaModel, Tensor};

    #[test]
    fn test_exit_node_is_vocab_projection() {
        // Create minimal model
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

        let mut model = LlamaModel::new(config.clone());

        // Add required weights
        model.tok_embeddings = Some(Tensor {
            data: vec![0.0f32; config.vocab_size * config.hidden_size],
            shape: vec![config.vocab_size, config.hidden_size],
            ..Default::default()
        });

        model.output = Some(Tensor {
            data: vec![0.0f32; config.vocab_size * config.hidden_size],
            shape: vec![config.vocab_size, config.hidden_size],
            ..Default::default()
        });

        model.norm = Some(Tensor {
            data: vec![1.0f32; config.hidden_size],
            shape: vec![config.hidden_size],
            ..Default::default()
        });

        // Add minimal layer weights
        let mut layer = LlamaLayer::default();
        layer.attention_norm = Some(Tensor {
            data: vec![1.0f32; config.hidden_size],
            shape: vec![config.hidden_size],
            ..Default::default()
        });
        layer.ffn_norm = Some(Tensor {
            data: vec![1.0f32; config.hidden_size],
            shape: vec![config.hidden_size],
            ..Default::default()
        });

        // Add dummy Q/K/V weights
        layer.wq = Some(Tensor {
            data: vec![0.0f32; config.hidden_size * config.hidden_size],
            shape: vec![config.hidden_size, config.hidden_size],
            ..Default::default()
        });
        layer.wk = Some(Tensor {
            data: vec![0.0f32; config.hidden_size * config.hidden_size],
            shape: vec![config.hidden_size, config.hidden_size],
            ..Default::default()
        });
        layer.wv = Some(Tensor {
            data: vec![0.0f32; config.hidden_size * config.hidden_size],
            shape: vec![config.hidden_size, config.hidden_size],
            ..Default::default()
        });
        layer.wo = Some(Tensor {
            data: vec![0.0f32; config.hidden_size * config.hidden_size],
            shape: vec![config.hidden_size, config.hidden_size],
            ..Default::default()
        });

        // Add FFN weights
        layer.w1 = Some(Tensor {
            data: vec![0.0f32; config.hidden_size * config.intermediate_size],
            shape: vec![config.intermediate_size, config.hidden_size],
            ..Default::default()
        });
        layer.w2 = Some(Tensor {
            data: vec![0.0f32; config.intermediate_size * config.hidden_size],
            shape: vec![config.hidden_size, config.intermediate_size],
            ..Default::default()
        });
        layer.w3 = Some(Tensor {
            data: vec![0.0f32; config.hidden_size * config.intermediate_size],
            shape: vec![config.intermediate_size, config.hidden_size],
            ..Default::default()
        });

        model.layers.push(layer);

        // Build DAG
        let dag = DagBuilder::new().build_model_dag(&model);

        // Verify exit point exists
        assert!(!dag.exit_points.is_empty(), "DAG should have exit points");

        let exit_node_id = dag.exit_points[0];
        let exit_node = &dag.nodes[exit_node_id];

        // Exit node should be either Gemv or FusedNormLinear
        match &exit_node.op {
            Operation::Gemv { out_size, .. } => {
                // Verify output size is vocab size
                assert_eq!(
                    *out_size, config.vocab_size,
                    "Exit Gemv output size should be vocab size"
                );
            }
            Operation::FusedNormLinear { out_size, .. } => {
                // Verify output size is vocab size
                assert_eq!(
                    *out_size, config.vocab_size,
                    "Exit FusedNormLinear output size should be vocab size"
                );
            }
            _ => {
                panic!(
                    "Exit node should be Gemv or FusedNormLinear, got {:?}",
                    std::mem::discriminant(&exit_node.op)
                );
            }
        }
    }
}
