// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG Executor - Traverse DAG and dispatch operations
//!
//! Executes operations by traversing optimized DAG nodes

use super::graph::{DagNode, Operation, OptimizedDag};
use crate::model::LlamaModel;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::buffer::ForwardBufferPool;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::inference::KVCacheGPU;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::cell::RefCell;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::collections::HashMap;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer as MetalBuffer;

/// Tensor registry: maps tensor IDs to GPU buffers
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct TensorRegistry {
    buffers: HashMap<usize, Rc<MetalBuffer>>,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl TensorRegistry {
    pub fn new() -> Self {
        Self { buffers: HashMap::new() }
    }

    pub fn get(&self, id: usize) -> Option<Rc<MetalBuffer>> {
        self.buffers.get(&id).cloned()
    }

    pub fn insert(&mut self, id: usize, buffer: Rc<MetalBuffer>) {
        self.buffers.insert(id, buffer);
    }
}

/// Execution context for DAG traversal
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct ExecutionContext<'a> {
    pub model: &'a LlamaModel,
    pub kv_cache: &'a mut KVCacheGPU,
    pub gpu_ctx: &'a crate::quant::vec_dot::gpu::metal::MetalContext,
    pub tensor_registry: TensorRegistry,
    pub weight_bindings: &'a HashMap<usize, crate::model::Tensor>,
    pub forward_pool: &'a mut ForwardBufferPool,
    pub layer_idx: usize,
    pub position: usize,
    pub use_mma: bool,    // Use MMA kernel for Flash Attention
    pub use_mma_v2: bool, // Use MMA v2 kernel
}

pub struct DagExecutor;

impl DagExecutor {
    /// Execute forward pass by traversing DAG nodes
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_dag(
        model: &LlamaModel,
        tokens: &[u32],
        kv_cache: &mut KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || model.tok_embeddings.is_none() {
            return None;
        }

        let dag = model.optimized_dag.as_ref()?;

        let hidden_size = model.config.hidden_size;
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Create buffer pool
        let reuse_pool: Rc<RefCell<Vec<Rc<MetalBuffer>>>> =
            Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        // Process first token (simplified - handle single token for now)
        let token = tokens[0];

        // 1. Embedding lookup
        let hidden = model.embedding_lookup_impl(token)?;
        let hidden_buffer = Rc::new(device.new_buffer_with_data(
            hidden.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Create tensor registry
        let mut tensor_registry = TensorRegistry::new();

        // Bind initial input tensor
        tensor_registry.insert(0, hidden_buffer);

        // Bind all weight tensors to GPU buffers
        for (tensor_id, tensor) in &dag.dag.weight_bindings {
            let weight_buffer = gpu_ctx.get_or_create_buffer(tensor);
            tensor_registry.insert(*tensor_id, weight_buffer);
        }

        // Check if MMA kernel is enabled via environment variable
        let use_mma = std::env::var("INFER_USE_MMA")
            .map(|v| v == "1" || v == "true" || v == "v2")
            .unwrap_or(false);
        let use_mma_v2 =
            std::env::var("INFER_USE_MMA").map(|v| v == "v2").unwrap_or(false);

        // Create execution context
        let mut exec_ctx = ExecutionContext {
            model,
            kv_cache,
            gpu_ctx,
            tensor_registry,
            weight_bindings: &dag.dag.weight_bindings,
            forward_pool: &mut forward_pool,
            layer_idx: 0,
            position: 0,
            use_mma,
            use_mma_v2,
        };

        // Track layer boundaries for committing command buffers
        // Each layer's FlashAttention has a unique layer_idx
        // Commit after completing each layer to avoid NaN issues
        let mut last_layer_idx: Option<usize> = None;
        let mut command_buffer = queue.new_command_buffer();
        let mut encoder = command_buffer.new_compute_command_encoder();

        // Traverse DAG nodes in execution order
        for node_id in &dag.execution_order {
            let node = &dag.dag.nodes[*node_id];

            // Debug: show registry state before executing node 534
            if std::env::var("INFER_DEBUG_DAG").is_ok() && *node_id == 534 {
                let keys: Vec<_> =
                    exec_ctx.tensor_registry.buffers.keys().collect();
                eprintln!(
                    "[DAG] Before node 534: registry has {} entries",
                    keys.len()
                );

                // Show the actual FusedQKV operation details
                if let Operation::FusedQKV {
                    input,
                    q_weight,
                    k_weight,
                    v_weight,
                    output_q,
                    output_k,
                    output_v,
                    ..
                } = &node.op
                {
                    eprintln!(
                        "[DAG] FusedQKV details: input={}, outputs={}/{}/{}",
                        input, output_q, output_k, output_v
                    );
                }
            }

            // Check if this operation belongs to a new layer
            let node_layer_idx = Self::get_node_layer_idx(node);
            let is_new_layer = node_layer_idx.is_some()
                && last_layer_idx.is_some()
                && node_layer_idx != last_layer_idx;

            if std::env::var("INFER_DEBUG_DAG").is_ok() {
                if node_layer_idx != last_layer_idx {
                    eprintln!(
                        "[DAG] Node {} layer_idx: {:?} (last: {:?}), is_new_layer: {}",
                        node_id, node_layer_idx, last_layer_idx, is_new_layer
                    );
                }
            }

            // If new layer, commit previous command buffer and start new one
            if is_new_layer {
                if std::env::var("INFER_DEBUG_DAG").is_ok() {
                    eprintln!(
                        "[DAG] Layer boundary at node {}, committing CB",
                        node_id
                    );
                }
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Start new command buffer for next layer
                command_buffer = queue.new_command_buffer();
                encoder = command_buffer.new_compute_command_encoder();
            }

            // Execute node
            let result = Self::execute_node(node, &mut exec_ctx, &encoder);
            if result.is_none() {
                if std::env::var("INFER_DEBUG_DAG").is_ok() {
                    eprintln!(
                        "[DAG] Node {} ({:?}) failed! Op details: {:?}",
                        node_id,
                        std::mem::discriminant(&node.op),
                        node.op
                    );
                    // Check which tensors are missing
                    match &node.op {
                        Operation::FusedGateUp {
                            input,
                            output_gate,
                            output_up,
                            ..
                        } => {
                            eprintln!(
                                "[DAG] FusedGateUp needs tensors: input={}, outputs={}/{}",
                                input, output_gate, output_up
                            );
                            if exec_ctx.tensor_registry.get(*input).is_none() {
                                eprintln!(
                                    "[DAG] Missing input tensor {}",
                                    input
                                );
                            }
                        }
                        Operation::FusedSwiGLU { gate, up, .. } => {
                            eprintln!(
                                "[DAG] FusedSwiGLU needs tensors: gate={}, up={}",
                                gate, up
                            );
                            if exec_ctx.tensor_registry.get(*gate).is_none() {
                                eprintln!("[DAG] Missing gate tensor {}", gate);
                            }
                            if exec_ctx.tensor_registry.get(*up).is_none() {
                                eprintln!("[DAG] Missing up tensor {}", up);
                            }
                        }
                        Operation::FusedAddNorm {
                            lhs,
                            rhs,
                            norm_weight,
                            ..
                        } => {
                            eprintln!(
                                "[DAG] FusedAddNorm needs tensors: lhs={}, rhs={}, norm_weight={}",
                                lhs, rhs, norm_weight
                            );
                            if exec_ctx.tensor_registry.get(*lhs).is_none() {
                                eprintln!("[DAG] Missing lhs tensor {}", lhs);
                            }
                            if exec_ctx.tensor_registry.get(*rhs).is_none() {
                                eprintln!("[DAG] Missing rhs tensor {}", rhs);
                            }
                            if exec_ctx
                                .tensor_registry
                                .get(*norm_weight)
                                .is_none()
                            {
                                eprintln!(
                                    "[DAG] Missing norm_weight tensor {}",
                                    norm_weight
                                );
                            }
                        }
                        _ => {}
                    }
                }
                // Clean up encoder before returning
                encoder.end_encoding();
                return None;
            }

            // Update last layer index
            if let Some(idx) = node_layer_idx {
                last_layer_idx = Some(idx);
            }
        }

        // Commit final command buffer
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Get output tensor
        let output_id = dag.dag.exit_points[0];
        let output_buffer = exec_ctx.tensor_registry.get(output_id)?;

        // Read logits
        let output_tensor = model.output.as_ref()?;
        let vocab_size = output_tensor.shape[0];
        let mut logits = vec![0.0f32; vocab_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                logits.as_mut_ptr(),
                vocab_size,
            );
        }

        Some(logits)
    }

    /// Execute a single DAG node
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn execute_node(
        node: &DagNode,
        ctx: &mut ExecutionContext,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Option<()> {
        match &node.op {
            Operation::EmbeddingLookup {
                token_id: _,
                embedding_weight: _,
                output: _,
            } => {
                // Embedding lookup already done before DAG traversal
                // This node is just a placeholder
                Some(())
            }

            Operation::RmsNorm { input, weight, output, epsilon } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let weight_buf = ctx.tensor_registry.get(*weight)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                ctx.gpu_ctx.rms_norm_encode(
                    encoder,
                    &input_buf,
                    &weight_buf,
                    &output_buf,
                    1,
                    ctx.model.config.hidden_size,
                    *epsilon,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::Gemv { input, weight, output, out_size } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let weight_buf = ctx.tensor_registry.get(*weight)?;
                let output_buf = ctx.forward_pool.get(*out_size * 4);

                // Get weight tensor dtype from weight_bindings
                let weight_tensor = ctx.weight_bindings.get(weight)?;
                let dtype = weight_tensor.dtype;

                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &weight_buf,
                    &input_buf,
                    &output_buf,
                    *out_size,
                    ctx.model.config.hidden_size,
                    dtype,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
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
                layer_idx: _,
            } => {
                if std::env::var("INFER_DEBUG_DAG").is_ok() {
                    eprintln!(
                        "[DAG] FusedQKV: input={}, q_weight={}, k_weight={}, v_weight={}",
                        input, q_weight, k_weight, v_weight
                    );
                }

                let input_buf = match ctx.tensor_registry.get(*input) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: input {} not in tensor_registry",
                                input
                            );
                        }
                        return None;
                    }
                };

                // Get weight buffers - check each one
                let q_weight_buf = match ctx.tensor_registry.get(*q_weight) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: q_weight {} not in tensor_registry",
                                q_weight
                            );
                        }
                        return None;
                    }
                };
                let k_weight_buf = match ctx.tensor_registry.get(*k_weight) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: k_weight {} not in tensor_registry",
                                k_weight
                            );
                        }
                        return None;
                    }
                };
                let v_weight_buf = match ctx.tensor_registry.get(*v_weight) {
                    Some(buf) => buf,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: v_weight {} not in tensor_registry",
                                v_weight
                            );
                        }
                        return None;
                    }
                };

                // Allocate output buffers
                let q_buf = ctx.forward_pool.get(*q_out_size * 4);
                let k_buf = ctx.forward_pool.get(*k_out_size * 4);
                let v_buf = ctx.forward_pool.get(*v_out_size * 4);

                // Get weight dtypes - check each one
                let q_dtype = match ctx.weight_bindings.get(q_weight) {
                    Some(t) => t.dtype,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: q_weight {} not in weight_bindings",
                                q_weight
                            );
                        }
                        return None;
                    }
                };
                let k_dtype = match ctx.weight_bindings.get(k_weight) {
                    Some(t) => t.dtype,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: k_weight {} not in weight_bindings",
                                k_weight
                            );
                        }
                        return None;
                    }
                };
                let v_dtype = match ctx.weight_bindings.get(v_weight) {
                    Some(t) => t.dtype,
                    None => {
                        if std::env::var("INFER_DEBUG_DAG").is_ok() {
                            eprintln!(
                                "[DAG] FusedQKV: v_weight {} not in weight_bindings",
                                v_weight
                            );
                        }
                        return None;
                    }
                };

                let hidden_size = ctx.model.config.hidden_size;

                // Run three GEMV operations
                // TODO: Optimize to single GEMM when weights are contiguous
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &q_weight_buf,
                    &input_buf,
                    &q_buf,
                    *q_out_size,
                    hidden_size,
                    q_dtype,
                );
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &k_weight_buf,
                    &input_buf,
                    &k_buf,
                    *k_out_size,
                    hidden_size,
                    k_dtype,
                );
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &v_weight_buf,
                    &input_buf,
                    &v_buf,
                    *v_out_size,
                    hidden_size,
                    v_dtype,
                );

                ctx.tensor_registry.insert(*output_q, q_buf);
                ctx.tensor_registry.insert(*output_k, k_buf);
                ctx.tensor_registry.insert(*output_v, v_buf);
                Some(())
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
                let input_buf = ctx.tensor_registry.get(*input)?;
                let gate_weight_buf = ctx.tensor_registry.get(*gate_weight)?;
                let up_weight_buf = ctx.tensor_registry.get(*up_weight)?;

                let gate_buf = ctx.forward_pool.get(*intermediate_size * 4);
                let up_buf = ctx.forward_pool.get(*intermediate_size * 4);

                let hidden_size = ctx.model.config.hidden_size;

                // Get weight dtypes
                let gate_dtype = ctx.weight_bindings.get(gate_weight)?.dtype;
                let up_dtype = ctx.weight_bindings.get(up_weight)?.dtype;

                // Run two GEMV operations
                // TODO: Optimize to single GEMM with concatenated weights
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &gate_weight_buf,
                    &input_buf,
                    &gate_buf,
                    *intermediate_size,
                    hidden_size,
                    gate_dtype,
                );
                ctx.gpu_ctx.mv_encode(
                    encoder,
                    &up_weight_buf,
                    &input_buf,
                    &up_buf,
                    *intermediate_size,
                    hidden_size,
                    up_dtype,
                );

                ctx.tensor_registry.insert(*output_gate, gate_buf);
                ctx.tensor_registry.insert(*output_up, up_buf);
                Some(())
            }

            Operation::RoPE { input, output, pos: _, n_heads, head_dim } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let output_buf = ctx.forward_pool.get(*n_heads * *head_dim * 4);

                // Create position slice
                let pos_slice = [ctx.position as i32];

                ctx.gpu_ctx.rope_neox_encode(
                    encoder,
                    &input_buf,
                    &pos_slice,
                    1, // n_seqs
                    *n_heads,
                    *head_dim,
                    10000.0, // freq_base
                    1.0,     // freq_scale
                    &output_buf,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FlashAttention {
                q,
                output,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            } => {
                let q_buf = ctx.tensor_registry.get(*q)?;

                // Get KV cache buffers directly from kv_cache object
                let k_buf = ctx.kv_cache.k_buffer(*layer_idx);
                let v_buf = ctx.kv_cache.v_buffer(*layer_idx);
                let output_buf = ctx.forward_pool.get(*n_heads * *head_dim * 4);

                // Get actual sequence length from kv_cache
                let seq_len = ctx.kv_cache.seq_len() + 1; // +1 for current token
                let kv_stride = ctx.kv_cache.max_seq();

                // Use MMA kernel if enabled and head_dim is 128
                if ctx.use_mma && *head_dim == 128 {
                    if ctx.use_mma_v2 {
                        ctx.gpu_ctx.flash_attn_mma_v2_encode(
                            encoder,
                            *n_heads,
                            *n_kv_heads,
                            seq_len,
                            kv_stride,
                            &q_buf,
                            k_buf,
                            v_buf,
                            &output_buf,
                        );
                    } else {
                        ctx.gpu_ctx.flash_attn_mma_encode(
                            encoder,
                            *n_heads,
                            *n_kv_heads,
                            seq_len,
                            kv_stride,
                            &q_buf,
                            k_buf,
                            v_buf,
                            &output_buf,
                        );
                    }
                } else {
                    ctx.gpu_ctx.flash_attn_f16_encode(
                        encoder,
                        1, // n_seqs
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        seq_len,
                        kv_stride,
                        &q_buf,
                        k_buf,
                        v_buf,
                        &output_buf,
                    );
                }

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::KVCacheUpdate { input, layer_idx, is_key } => {
                let input_buf = ctx.tensor_registry.get(*input)?;

                // Get model config
                let n_kv_heads = ctx
                    .model
                    .layers
                    .get(*layer_idx)
                    .and_then(|l| l.wk.as_ref())
                    .map(|w| w.shape[1] / ctx.model.config.head_dim)
                    .unwrap_or(ctx.model.config.n_heads);
                let head_dim = ctx.model.config.head_dim;
                let pos = ctx.kv_cache.seq_len();
                let max_seq = ctx.kv_cache.max_seq();

                if *is_key {
                    // Update K cache
                    let k_cache_buf = ctx.kv_cache.k_buffer(*layer_idx);
                    ctx.gpu_ctx.kv_update_k_encode(
                        encoder,
                        &input_buf,
                        k_cache_buf,
                        pos,
                        max_seq,
                        n_kv_heads,
                        head_dim,
                    );
                } else {
                    // Update V cache
                    let v_cache_buf = ctx.kv_cache.v_buffer(*layer_idx);
                    ctx.gpu_ctx.kv_update_v_encode(
                        encoder,
                        &input_buf,
                        v_cache_buf,
                        pos,
                        max_seq,
                        n_kv_heads,
                        head_dim,
                    );
                }

                Some(())
            }

            Operation::SiLU { input, output } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                ctx.gpu_ctx.silu_encode(
                    encoder,
                    &input_buf,
                    &output_buf,
                    ctx.model.config.hidden_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::Mul { lhs, rhs, output } => {
                let lhs_buf = ctx.tensor_registry.get(*lhs)?;
                let rhs_buf = ctx.tensor_registry.get(*rhs)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                ctx.gpu_ctx.mul_encode(
                    encoder,
                    &lhs_buf,
                    &rhs_buf,
                    &output_buf,
                    ctx.model.config.hidden_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::Add { lhs, rhs, output } => {
                let lhs_buf = ctx.tensor_registry.get(*lhs)?;
                let rhs_buf = ctx.tensor_registry.get(*rhs)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                ctx.gpu_ctx.add_encode(
                    encoder,
                    &lhs_buf,
                    &rhs_buf,
                    &output_buf,
                    ctx.model.config.hidden_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            // Fused operations
            Operation::FusedSwiGLU { gate, up, output, layer_idx: _ } => {
                let gate_buf = ctx.tensor_registry.get(*gate)?;
                let up_buf = ctx.tensor_registry.get(*up)?;
                let output_buf =
                    ctx.forward_pool.get(ctx.model.config.hidden_size * 4);

                ctx.gpu_ctx.fused_swiglu_encode(
                    encoder,
                    &gate_buf,
                    &up_buf,
                    &output_buf,
                    ctx.model.config.hidden_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FusedNormLinear {
                input,
                norm_weight,
                linear_weight,
                output,
                out_size,
                in_size: _,
                epsilon,
                layer_idx: _,
            } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let norm_weight_buf = ctx.tensor_registry.get(*norm_weight)?;
                let linear_weight_buf =
                    ctx.tensor_registry.get(*linear_weight)?;
                let output_buf = ctx.forward_pool.get(*out_size * 4);

                // Get weight tensor info
                let weight_tensor = ctx.weight_bindings.get(linear_weight)?;

                // Use F16 fused kernel for F16 weights, fallback otherwise
                match weight_tensor.dtype {
                    crate::model::tensor::DataType::F16 => {
                        ctx.gpu_ctx.fused_norm_linear_f16_encode(
                            encoder,
                            &input_buf,
                            &norm_weight_buf,
                            &linear_weight_buf,
                            &output_buf,
                            *out_size,
                            ctx.model.config.hidden_size,
                            *epsilon,
                        );
                    }
                    _ => {
                        // Fallback: separate RMSNorm + GEMV for other dtypes
                        // First, do RMSNorm
                        let norm_out_buf = ctx
                            .forward_pool
                            .get(ctx.model.config.hidden_size * 4);
                        ctx.gpu_ctx.rms_norm_encode(
                            encoder,
                            &input_buf,
                            &norm_weight_buf,
                            &norm_out_buf,
                            1,
                            ctx.model.config.hidden_size,
                            *epsilon,
                        );

                        // Then, do GEMV
                        ctx.gpu_ctx.mv_encode(
                            encoder,
                            &linear_weight_buf,
                            &norm_out_buf,
                            &output_buf,
                            *out_size,
                            ctx.model.config.hidden_size,
                            weight_tensor.dtype,
                        );
                    }
                }

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FusedAddNorm {
                lhs,
                rhs,
                norm_weight,
                output,
                epsilon,
                layer_idx: _,
            } => {
                let lhs_buf = ctx.tensor_registry.get(*lhs)?;
                let rhs_buf = ctx.tensor_registry.get(*rhs)?;
                let norm_weight_buf = ctx.tensor_registry.get(*norm_weight)?;

                let hidden_size = ctx.model.config.hidden_size;
                let output_buf = ctx.forward_pool.get(hidden_size * 4);

                // Use fused kernel
                ctx.gpu_ctx.fused_add_norm_encode(
                    encoder,
                    &lhs_buf,
                    &rhs_buf,
                    &norm_weight_buf,
                    &output_buf,
                    hidden_size,
                    *epsilon,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::FusedGemvAdd {
                input,
                weight,
                residual,
                output,
                out_size,
                in_size,
                layer_idx: _,
            } => {
                let input_buf = ctx.tensor_registry.get(*input)?;
                let weight_tensor = ctx.weight_bindings.get(weight)?;
                let residual_buf = ctx.tensor_registry.get(*residual)?;

                // Output buffer - GEMV result + residual add
                let output_buf = ctx.forward_pool.get(*out_size * 4);

                // Step 1: Execute GEMV to output buffer
                ctx.model.gemv_encode(
                    encoder,
                    &input_buf,
                    *in_size,
                    weight_tensor,
                    *out_size,
                    ctx.gpu_ctx,
                    &output_buf,
                )?;

                // Step 2: Add residual to output buffer in-place
                ctx.gpu_ctx.add_encode(
                    encoder,
                    &residual_buf,
                    &output_buf,
                    &output_buf,
                    *out_size,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                #[cfg(debug_assertions)]
                eprintln!(
                    "[FusedGemvAdd] input={}, weight={}, residual={}, output={}, out_size={}",
                    input, weight, residual, output, out_size
                );
                Some(())
            }

            Operation::ExpertRouting { .. } => {
                // MoE - not implemented yet
                None
            }

            Operation::ExpertCompute { .. } => {
                // MoE - not implemented yet
                None
            }

            Operation::NoOp => {
                // Placeholder for fused/removed operations - skip
                Some(())
            }
        }
    }

    /// Get layer index from DAG node (if it belongs to a specific layer)
    fn get_node_layer_idx(node: &DagNode) -> Option<usize> {
        // Use the node's layer_idx field
        node.layer_idx
    }
}
