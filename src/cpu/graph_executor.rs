// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Graph-level CPU dispatch - all threads traverse DAG together

use crate::dag::Operation;
use crate::inference::KVCacheBase;
use crate::model::Tensor;
use crate::model::llama::LlamaModel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Shared graph execution state
pub struct GraphExecutionState {
    pub model: *const LlamaModel,
    pub kv_cache: *mut crate::inference::KVCache,
    pub dag: *const crate::dag::OptimizedDag,
    pub weights: *const HashMap<usize, Tensor>,
    pub token: u32,
    pub position: usize,
    pub output: Arc<Mutex<Option<Vec<f32>>>>,
    /// Shared tensor storage
    pub shared_tensors: Arc<Mutex<Vec<Vec<f32>>>>,
}

unsafe impl Send for GraphExecutionState {}
unsafe impl Sync for GraphExecutionState {}

/// Execute entire DAG with all threads
pub fn execute_graph_parallel(
    model: &LlamaModel,
    token: u32,
    position: usize,
    kv_cache: &mut crate::inference::KVCache,
    dag: &crate::dag::OptimizedDag,
    weights: &HashMap<usize, Tensor>,
) -> Option<Vec<f32>> {
    let state = Box::new(GraphExecutionState {
        model: model as *const _,
        kv_cache: kv_cache as *mut _,
        dag: dag as *const _,
        weights: weights as *const _,
        token,
        position,
        output: Arc::new(Mutex::new(None)),
        shared_tensors: Arc::new(Mutex::new(vec![Vec::new(); 1024])),
    });

    let state_ptr = Box::into_raw(state);

    crate::cpu::thread_pool::GLOBAL_THREAD_POOL.execute_graph(
        graph_task_fn,
        state_ptr as usize,
        0,
    );

    let state = unsafe { Box::from_raw(state_ptr) };
    state.output.lock().unwrap().take()
}

/// Task function for graph execution
unsafe fn graph_task_fn(
    ith: usize,
    nth: usize,
    state_ptr: usize,
    _user_data: usize,
) {
    let state = &*(state_ptr as *const GraphExecutionState);
    let model = &*state.model;
    let kv_cache = &mut *state.kv_cache;
    let dag = &*state.dag;
    let weights = &*state.weights;
    let barrier = crate::cpu::thread_pool::GLOBAL_THREAD_POOL.barrier();

    for (_idx, node_id) in dag.execution_order.iter().enumerate() {
        let node = &dag.dag.nodes[*node_id];

        match &node.op {
            Operation::EmbeddingLookup { output, .. } => {
                if ith == 0 {
                    if let Some(embed) =
                        model.embedding_lookup_impl(state.token)
                    {
                        let mut tensors = state.shared_tensors.lock().unwrap();
                        tensors[*output].resize(embed.len(), 0.0);
                        tensors[*output][..].copy_from_slice(&embed);
                    }
                }
            }

            Operation::RmsNorm { input, weight, output, .. } => {
                if ith == 0 {
                    let input_data =
                        state.shared_tensors.lock().unwrap()[*input].clone();
                    if let Some(w) = weights.get(weight) {
                        let mut tensors = state.shared_tensors.lock().unwrap();
                        tensors[*output].resize(input_data.len(), 0.0);
                        model.apply_rms_norm_into(
                            &input_data,
                            w,
                            &mut tensors[*output],
                        );
                    }
                }
            }

            Operation::Gemv { input, weight, output, out_size, is_transposed } => {
                let input_data = {
                    let tensors = state.shared_tensors.lock().unwrap();
                    if *input < tensors.len() {
                        tensors[*input].clone()
                    } else {
                        Vec::new()
                    }
                };

                let m = *out_size;
                let k = input_data.len();

                // Thread 0 allocates output tensor
                if ith == 0 {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    tensors[*output].resize(m, 0.0);
                }

                // ALL threads must reach barrier before any GEMV computation
                barrier.wait();

                if let Some(w) = weights.get(weight) {
                    if *is_transposed {
                        // Transposed GEMV: thread 0 computes full result
                        // Weight shape is [k, m], need to compute y = W^T * x
                        if ith == 0 {
                            let result = model.gemv_transposed(&input_data, w, m);
                            if let Some(r) = result {
                                let mut tensors = state.shared_tensors.lock().unwrap();
                                tensors[*output] = r;
                            }
                        }
                    } else {
                        // Parallel GEMV: all threads participate
                        let mut tensors = state.shared_tensors.lock().unwrap();
                        let out_data = &mut tensors[*output];
                        model.gemv_parallel_row(
                            ith,
                            nth,
                            &input_data,
                            w,
                            out_data,
                            k,
                            m,
                        );
                    }
                } else if ith == 0 {
                    eprintln!(
                        "[GraphExecutor] Weight {} not found for GEMV!",
                        weight
                    );
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
                let input_data = {
                    let tensors = state.shared_tensors.lock().unwrap();
                    if *input < tensors.len() {
                        tensors[*input].clone()
                    } else {
                        Vec::new()
                    }
                };
                let k = input_data.len();

                if ith == 0 {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    tensors[*output_q].resize(*q_out_size, 0.0);
                    tensors[*output_k].resize(*k_out_size, 0.0);
                    tensors[*output_v].resize(*v_out_size, 0.0);
                }
                barrier.wait();

                if let Some(q_w) = weights.get(q_weight) {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    let out = &mut tensors[*output_q];
                    model.gemv_parallel_row(
                        ith,
                        nth,
                        &input_data,
                        q_w,
                        out,
                        k,
                        *q_out_size,
                    );
                }
                barrier.wait();

                if let Some(k_w) = weights.get(k_weight) {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    let out = &mut tensors[*output_k];
                    model.gemv_parallel_row(
                        ith,
                        nth,
                        &input_data,
                        k_w,
                        out,
                        k,
                        *k_out_size,
                    );
                }
                barrier.wait();

                if let Some(v_w) = weights.get(v_weight) {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    let out = &mut tensors[*output_v];
                    model.gemv_parallel_row(
                        ith,
                        nth,
                        &input_data,
                        v_w,
                        out,
                        k,
                        *v_out_size,
                    );
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
                // Fused QKV with merged weight: single GEMV, split result
                let input_data = {
                    let tensors = state.shared_tensors.lock().unwrap();
                    if *input < tensors.len() {
                        tensors[*input].clone()
                    } else {
                        Vec::new()
                    }
                };
                let k = input_data.len();
                let total_out_size = q_out_size + k_out_size + v_out_size;

                // Allocate temporary buffer in the LAST tensor slot to avoid collision
                let temp_id = 1023;

                // Thread 0 allocates output tensors
                if ith == 0 {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    tensors[*output_q].resize(*q_out_size, 0.0);
                    tensors[*output_k].resize(*k_out_size, 0.0);
                    tensors[*output_v].resize(*v_out_size, 0.0);
                    tensors[temp_id].resize(total_out_size, 0.0);
                }
                barrier.wait();

                // Do single GEMV into temp buffer
                if let Some(w) = weights.get(qkv_weight) {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    let qkv_out = &mut tensors[temp_id];
                    model.gemv_parallel_row(
                        ith,
                        nth,
                        &input_data,
                        w,
                        qkv_out,
                        k,
                        total_out_size,
                    );
                }
                barrier.wait();

                // Thread 0 splits the result
                if ith == 0 {
                    let tensors = state.shared_tensors.lock().unwrap();
                    let qkv_out = &tensors[temp_id];

                    // Split into Q, K, V
                    let q_data = qkv_out[0..*q_out_size].to_vec();
                    let k_data = qkv_out
                        [*q_out_size..*q_out_size + *k_out_size]
                        .to_vec();
                    let v_data = qkv_out
                        [*q_out_size + *k_out_size..total_out_size]
                        .to_vec();

                    // We need mutable access to write, drop the immutable borrow first
                    drop(tensors);
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    tensors[*output_q] = q_data;
                    tensors[*output_k] = k_data;
                    tensors[*output_v] = v_data;
                }
            }

            Operation::RoPE { input, output, n_heads, head_dim, .. } => {
                if ith == 0 {
                    let input_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *input < tensors.len() {
                            tensors[*input].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    tensors[*output].resize(input_data.len(), 0.0);
                    let _ = model.apply_rope_into(
                        &input_data,
                        state.position,
                        *n_heads,
                        *head_dim,
                        &mut tensors[*output],
                    );
                }
            }

            Operation::KVCacheUpdate {
                input,
                layer_idx,
                is_key,
                n_kv_heads,
                head_dim,
            } => {
                if ith == 0 {
                    let input_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *input < tensors.len() {
                            tensors[*input].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    if *is_key {
                        kv_cache.set_k(
                            *layer_idx,
                            &input_data,
                            *n_kv_heads,
                            *head_dim,
                        );
                    } else {
                        kv_cache.set_v(
                            *layer_idx,
                            &input_data,
                            *n_kv_heads,
                            *head_dim,
                        );
                    }
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
                if ith == 0 {
                    let q_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *q < tensors.len() {
                            tensors[*q].clone()
                        } else {
                            Vec::new()
                        }
                    };
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
                                let v_s =
                                    kv_cache.get_v(*layer_idx, kv_head, s);
                                sum += scores[s] * v_s[d];
                            }
                            out_data[h * head_dim + d] = sum;
                        }
                    }
                    state.shared_tensors.lock().unwrap()[*output] = out_data;
                }
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
                let input_data = {
                    let tensors = state.shared_tensors.lock().unwrap();
                    if *input < tensors.len() {
                        tensors[*input].clone()
                    } else {
                        Vec::new()
                    }
                };

                if ith == 0 {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    tensors[*output_gate].resize(*intermediate_size, 0.0);
                    tensors[*output_up].resize(*intermediate_size, 0.0);
                }
                barrier.wait();

                if let Some(gate_w) = weights.get(gate_weight) {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    let out = &mut tensors[*output_gate];
                    model.gemv_parallel_row(
                        ith,
                        nth,
                        &input_data,
                        gate_w,
                        out,
                        input_data.len(),
                        *intermediate_size,
                    );
                }
                barrier.wait();

                if let Some(up_w) = weights.get(up_weight) {
                    let mut tensors = state.shared_tensors.lock().unwrap();
                    let out = &mut tensors[*output_up];
                    model.gemv_parallel_row(
                        ith,
                        nth,
                        &input_data,
                        up_w,
                        out,
                        input_data.len(),
                        *intermediate_size,
                    );
                }
            }

            Operation::FusedSwiGLU { gate, up, output, .. } => {
                if ith == 0 {
                    let gate_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *gate < tensors.len() {
                            tensors[*gate].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let up_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *up < tensors.len() {
                            tensors[*up].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let len = gate_data.len().min(up_data.len());
                    let mut result = vec![0.0f32; len];
                    for i in 0..len {
                        // Correct: SiLU is applied to UP, not gate
                        // result = gate * silu(up)
                        let silu = up_data[i] / (1.0 + (-up_data[i]).exp());
                        result[i] = gate_data[i] * silu;
                    }
                    state.shared_tensors.lock().unwrap()[*output] = result;
                }
            }

            Operation::Add { lhs, rhs, output } => {
                if ith == 0 {
                    let lhs_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *lhs < tensors.len() {
                            tensors[*lhs].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let rhs_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *rhs < tensors.len() {
                            tensors[*rhs].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let len = lhs_data.len().min(rhs_data.len());
                    let mut result = vec![0.0f32; len];
                    for i in 0..len {
                        result[i] = lhs_data[i] + rhs_data[i];
                    }
                    state.shared_tensors.lock().unwrap()[*output] = result;
                }
            }

            Operation::SiLU { input, output } => {
                if ith == 0 {
                    let input_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *input < tensors.len() {
                            tensors[*input].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let mut result = vec![0.0f32; input_data.len()];
                    for i in 0..input_data.len() {
                        result[i] =
                            input_data[i] / (1.0 + (-input_data[i]).exp());
                    }
                    state.shared_tensors.lock().unwrap()[*output] = result;
                }
            }

            Operation::Mul { lhs, rhs, output } => {
                if ith == 0 {
                    let lhs_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *lhs < tensors.len() {
                            tensors[*lhs].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let rhs_data = {
                        let tensors = state.shared_tensors.lock().unwrap();
                        if *rhs < tensors.len() {
                            tensors[*rhs].clone()
                        } else {
                            Vec::new()
                        }
                    };
                    let len = lhs_data.len().min(rhs_data.len());
                    let mut result = vec![0.0f32; len];
                    for i in 0..len {
                        result[i] = lhs_data[i] * rhs_data[i];
                    }
                    state.shared_tensors.lock().unwrap()[*output] = result;
                }
            }

            Operation::NoOp => {}

            _ => {}
        }

        // Barrier after each node - ALL threads must reach this
        barrier.wait();
    }

    if ith == 0 {
        kv_cache.advance();
    }

    if ith == 0 {
        // Find output from the exit node (output projection)
        let mut found_output = false;
        for node_id in dag.execution_order.iter().rev() {
            let node = &dag.dag.nodes[*node_id];
            if let Operation::Gemv { output, out_size, .. } = &node.op {
                let tensors = state.shared_tensors.lock().unwrap();
                let out_tensor = &tensors[*output];
                if !out_tensor.is_empty() {
                    let result = out_tensor[..*out_size].to_vec();
                    *state.output.lock().unwrap() = Some(result);
                    found_output = true;
                } else {
                    eprintln!(
                        "[GraphExecutor] Output tensor {} is empty, out_size={}",
                        output, out_size
                    );
                }
                break;
            }
            if let Operation::Add { output, .. } = &node.op {
                let tensors = state.shared_tensors.lock().unwrap();
                if !tensors[*output].is_empty() {
                    let result = tensors[*output].clone();
                    *state.output.lock().unwrap() = Some(result);
                    found_output = true;
                }
                break;
            }
        }
        if !found_output {
            eprintln!("[GraphExecutor] Failed to find output!");
        }
    }
}
