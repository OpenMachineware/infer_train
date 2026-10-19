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
        };

        // Create command buffer for entire forward pass
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Traverse DAG nodes in execution order
        for node_id in &dag.execution_order {
            let node = &dag.dag.nodes[*node_id];
            Self::execute_node(node, &mut exec_ctx, encoder)?;
        }

        // Commit and wait
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
                k_cache,
                v_cache,
                output,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_seq_len,
            } => {
                let q_buf = ctx.tensor_registry.get(*q)?;
                let k_buf = ctx.tensor_registry.get(*k_cache)?;
                let v_buf = ctx.tensor_registry.get(*v_cache)?;
                let output_buf = ctx.forward_pool.get(*n_heads * *head_dim * 4);

                ctx.gpu_ctx.flash_attn_f16_encode(
                    encoder,
                    1, // n_seqs
                    *n_heads,
                    *n_kv_heads,
                    *head_dim,
                    1,           // seq_len = 1 for decode
                    *kv_seq_len, // kv_stride
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &output_buf,
                );

                ctx.tensor_registry.insert(*output, output_buf);
                Some(())
            }

            Operation::KVCacheUpdate { input: _, cache: _, pos: _ } => {
                // KV cache update logic
                // For now, skip - will be handled separately
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
            Operation::FusedSwiGLU { gate, up, output } => {
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
}
