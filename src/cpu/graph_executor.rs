// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Graph-level CPU dispatch - placeholder for future optimization
//!
//! Current status: Single-threaded execution via forward_pooled is the optimal
//! approach for decode due to memory bandwidth being the bottleneck.
//! The graph executor approach (multi-threaded DAG traversal) was found to be
//! 32x slower than single-threaded due to mutex contention.

use crate::dag::Operation;
use crate::inference::KVCacheBase;
use crate::model::Tensor;
use crate::model::llama::LlamaModel;
use std::collections::HashMap;

/// Pre-allocated buffer for graph execution
pub struct GraphWorkBuffers {
    /// Hidden states for each tensor ID
    pub tensors: Vec<Vec<f32>>,
}

impl GraphWorkBuffers {
    pub fn new(
        _hidden_size: usize,
        _intermediate_size: usize,
        _qkv_out_size: usize,
        max_tensors: usize,
    ) -> Self {
        let tensors = vec![Vec::new(); max_tensors];
        GraphWorkBuffers { tensors }
    }
}

/// Execute entire DAG - currently uses single-threaded fallback
/// Multi-threaded graph execution was found to be 32x slower due to
/// synchronization overhead. Single-threaded execution via forward_pooled
/// is optimal for decode due to memory bandwidth bottleneck.
pub fn execute_graph_parallel(
    model: &LlamaModel,
    token: u32,
    position: usize,
    kv_cache: &mut crate::inference::KVCache,
    dag: &crate::dag::OptimizedDag,
    weights: &HashMap<usize, Tensor>,
) -> Option<Vec<f32>> {
    // Fallback to single-threaded execution
    // This is actually the optimal approach for decode
    execute_graph_single_thread(model, token, position, kv_cache, dag, weights)
}

/// Single-threaded graph execution
fn execute_graph_single_thread(
    model: &LlamaModel,
    token: u32,
    _position: usize,
    kv_cache: &mut crate::inference::KVCache,
    dag: &crate::dag::OptimizedDag,
    weights: &HashMap<usize, Tensor>,
) -> Option<Vec<f32>> {
    let mut tensors: Vec<Vec<f32>> = vec![Vec::new(); 1024];

    for node_id in &dag.execution_order {
        let node = &dag.dag.nodes[*node_id];

        match &node.op {
            Operation::EmbeddingLookup { output, .. } => {
                if let Some(embed) = model.embedding_lookup_impl(token) {
                    tensors[*output] = embed;
                }
            }

            Operation::RmsNorm { input, weight, output, .. } => {
                let input_data = &tensors[*input];
                if let Some(w) = weights.get(weight) {
                    let mut result = vec![0.0f32; input_data.len()];
                    model.apply_rms_norm_into(input_data, w, &mut result);
                    tensors[*output] = result;
                }
            }

            Operation::Gemv {
                input,
                weight,
                output,
                out_size,
                is_transposed,
            } => {
                let input_data = &tensors[*input];
                let m = *out_size;

                if let Some(w) = weights.get(weight) {
                    if *is_transposed {
                        if let Some(result) =
                            model.gemv_transposed(input_data, w, m)
                        {
                            tensors[*output] = result;
                        }
                    } else {
                        let mut result = vec![0.0f32; m];
                        model.gemv_into(input_data, w, &mut result);
                        tensors[*output] = result;
                    }
                }
            }

            Operation::FusedQKV {
                input,
                q_weight,
                k_weight,
                v_weight,
                output_q,
                output_k,
                output_v,
                q_out_size,
                k_out_size,
                v_out_size,
                ..
            } => {
                let input_data = tensors[*input].clone();

                if let Some(q_w) = weights.get(q_weight) {
                    let mut q_out = vec![0.0f32; *q_out_size];
                    model.gemv_into(&input_data, q_w, &mut q_out);
                    tensors[*output_q] = q_out;
                }

                if let Some(k_w) = weights.get(k_weight) {
                    let mut k_out = vec![0.0f32; *k_out_size];
                    model.gemv_into(&input_data, k_w, &mut k_out);
                    tensors[*output_k] = k_out;
                }

                if let Some(v_w) = weights.get(v_weight) {
                    let mut v_out = vec![0.0f32; *v_out_size];
                    model.gemv_into(&input_data, v_w, &mut v_out);
                    tensors[*output_v] = v_out;
                }
            }

            Operation::FusedQKVWeight {
                input,
                qkv_weight,
                output_q,
                output_k,
                output_v,
                q_out_size,
                k_out_size,
                v_out_size,
                ..
            } => {
                let input_data = tensors[*input].clone();
                let total_out_size = q_out_size + k_out_size + v_out_size;

                if let Some(w) = weights.get(qkv_weight) {
                    let mut qkv_out = vec![0.0f32; total_out_size];
                    model.gemv_into(&input_data, w, &mut qkv_out);

                    tensors[*output_q] = qkv_out[..*q_out_size].to_vec();
                    tensors[*output_k] = qkv_out
                        [*q_out_size..*q_out_size + *k_out_size]
                        .to_vec();
                    tensors[*output_v] =
                        qkv_out[*q_out_size + *k_out_size..].to_vec();
                }
            }

            Operation::RoPE { input, output, n_heads, head_dim, .. } => {
                let input_data = &tensors[*input];
                let mut result = vec![0.0f32; input_data.len()];
                let pos = kv_cache.seq_len();
                let _ = model.apply_rope_into(
                    input_data,
                    pos,
                    *n_heads,
                    *head_dim,
                    &mut result,
                );
                tensors[*output] = result;
            }

            Operation::KVCacheUpdate {
                input,
                layer_idx,
                is_key,
                n_kv_heads,
                head_dim,
            } => {
                let input_data = &tensors[*input];
                if *is_key {
                    kv_cache.set_k(
                        *layer_idx,
                        input_data,
                        *n_kv_heads,
                        *head_dim,
                    );
                } else {
                    kv_cache.set_v(
                        *layer_idx,
                        input_data,
                        *n_kv_heads,
                        *head_dim,
                    );
                }
            }

            Operation::FlashAttention {
                q,
                output,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            } => {
                let q_data = &tensors[*q];
                let seq_len = kv_cache.seq_len() + 1;
                let out_size = *n_heads * head_dim;
                let scale = 1.0f32 / (*head_dim as f32).sqrt();

                let mut out_data = vec![0.0f32; out_size];
                for h in 0..*n_heads {
                    let kv_head = h * n_kv_heads / n_heads;
                    let mut scores = vec![0.0f32; seq_len];

                    for s in 0..seq_len {
                        let k_s = kv_cache.get_k(*layer_idx, kv_head, s);
                        let mut score = 0.0f32;
                        for d in 0..*head_dim {
                            score += q_data[h * head_dim + d] * k_s[d];
                        }
                        scores[s] = score * scale;
                    }

                    let max_score = scores
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in 0..seq_len {
                        scores[s] = (scores[s] - max_score).exp();
                        sum_exp += scores[s];
                    }
                    for s in 0..seq_len {
                        scores[s] /= sum_exp;
                    }

                    for d in 0..*head_dim {
                        let mut sum = 0.0f32;
                        for s in 0..seq_len {
                            let v_s = kv_cache.get_v(*layer_idx, kv_head, s);
                            sum += scores[s] * v_s[d];
                        }
                        out_data[h * head_dim + d] = sum;
                    }
                }
                tensors[*output] = out_data;
            }

            Operation::FusedGateUp {
                input,
                gate_weight,
                up_weight,
                output_gate,
                output_up,
                intermediate_size,
                ..
            } => {
                let input_data = tensors[*input].clone();

                if let Some(gate_w) = weights.get(gate_weight) {
                    let mut gate_out = vec![0.0f32; *intermediate_size];
                    model.gemv_into(&input_data, gate_w, &mut gate_out);
                    tensors[*output_gate] = gate_out;
                }

                if let Some(up_w) = weights.get(up_weight) {
                    let mut up_out = vec![0.0f32; *intermediate_size];
                    model.gemv_into(&input_data, up_w, &mut up_out);
                    tensors[*output_up] = up_out;
                }
            }

            Operation::FusedSwiGLU { gate, up, output, .. } => {
                let gate_data = &tensors[*gate];
                let up_data = &tensors[*up];
                let len = gate_data.len().min(up_data.len());

                let mut result = vec![0.0f32; len];
                for i in 0..len {
                    let silu = up_data[i] / (1.0 + (-up_data[i]).exp());
                    result[i] = gate_data[i] * silu;
                }
                tensors[*output] = result;
            }

            Operation::Add { lhs, rhs, output } => {
                let lhs_data = &tensors[*lhs];
                let rhs_data = &tensors[*rhs];
                let len = lhs_data.len().min(rhs_data.len());

                let mut result = vec![0.0f32; len];
                for i in 0..len {
                    result[i] = lhs_data[i] + rhs_data[i];
                }
                tensors[*output] = result;
            }

            Operation::SiLU { input, output } => {
                let input_data = &tensors[*input];
                let mut result = vec![0.0f32; input_data.len()];
                for i in 0..input_data.len() {
                    result[i] = input_data[i] / (1.0 + (-input_data[i]).exp());
                }
                tensors[*output] = result;
            }

            Operation::Mul { lhs, rhs, output } => {
                let lhs_data = &tensors[*lhs];
                let rhs_data = &tensors[*rhs];
                let len = lhs_data.len().min(rhs_data.len());

                let mut result = vec![0.0f32; len];
                for i in 0..len {
                    result[i] = lhs_data[i] * rhs_data[i];
                }
                tensors[*output] = result;
            }

            Operation::NoOp => {}

            _ => {}
        }
    }

    kv_cache.advance();

    // Find output from the exit node
    for node_id in dag.execution_order.iter().rev() {
        let node = &dag.dag.nodes[*node_id];
        match &node.op {
            Operation::Gemv { output, out_size, .. } => {
                if !tensors[*output].is_empty() {
                    return Some(tensors[*output][..*out_size].to_vec());
                }
                break;
            }
            Operation::Add { output, .. } => {
                if !tensors[*output].is_empty() {
                    return Some(tensors[*output].clone());
                }
                break;
            }
            _ => {}
        }
    }

    None
}
