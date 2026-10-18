// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG Executor - Traverse DAG and dispatch operations
//!
//! Executes operations in topological order with batched command buffers

use super::graph::{ComputeDag, DagNode, DevicePreference, Operation};
use crate::buffer::BufferManager;
use crate::compute::backend::Backend;
use crate::inference::KVCacheGPU;
use crate::model::LlamaModel;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{CommandBuffer, ComputeCommandEncoder};

pub struct DagExecutor {
    buffer_mgr: BufferManager,
}

impl DagExecutor {
    pub fn new(buffer_mgr: BufferManager) -> Self {
        Self { buffer_mgr }
    }

    /// Execute DAG on GPU with batched command buffers
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn execute_gpu_batched(
        &mut self,
        dag: &ComputeDag,
        model: &LlamaModel,
        kv_cache: &mut KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<std::rc::Rc<metal::Buffer>> {
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Single command buffer for entire DAG
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Get execution order
        let exec_order = dag.execution_order();

        // Track current buffers (tensor_id -> buffer)
        let mut tensor_buffers: std::collections::HashMap<
            usize,
            std::rc::Rc<metal::Buffer>,
        > = std::collections::HashMap::new();

        // Execute nodes in order
        for node_id in exec_order {
            let node = &dag.nodes[node_id];

            match &node.op {
                Operation::RmsNorm { input, weight, output, epsilon } => {
                    let input_buf = tensor_buffers.get(input)?;
                    let weight_buf =
                        self.buffer_mgr.get_or_create_buffer_gpu(weight)?;
                    let output_buf = gpu_ctx.rms_norm_encode(
                        encoder, input_buf, weight_buf, 1,
                        1024, // Will be set dynamically
                        *epsilon,
                    );
                    tensor_buffers.insert(*output, output_buf);
                }

                Operation::Gemv { input, weight, output, out_size } => {
                    let input_buf = tensor_buffers.get(input)?;
                    let weight_buf =
                        self.buffer_mgr.get_or_create_buffer_gpu(weight)?;

                    // Call GEMV kernel
                    let output_buf = self.gemv_encode(
                        encoder, input_buf, weight_buf, *out_size, gpu_ctx,
                    )?;
                    tensor_buffers.insert(*output, output_buf);
                }

                Operation::RoPE { input, output, pos, n_heads, head_dim } => {
                    let input_buf = tensor_buffers.get(input)?;
                    let output_buf = gpu_ctx.rope_neox_encode(
                        encoder,
                        input_buf,
                        &[*pos as i32],
                        1,
                        *n_heads,
                        *head_dim,
                        10000.0,
                        1.0,
                    );
                    tensor_buffers.insert(*output, output_buf);
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
                    let q_buf = tensor_buffers.get(q)?;
                    let k_cache_buf = kv_cache.k_buffer(0); // Will be set dynamically
                    let v_cache_buf = kv_cache.v_buffer(0);

                    let output_buf = gpu_ctx.flash_attn_f16_encode(
                        encoder,
                        1,
                        *n_heads,
                        *n_kv_heads,
                        *head_dim,
                        *kv_seq_len,
                        kv_cache.max_seq(),
                        q_buf,
                        k_cache_buf,
                        v_cache_buf,
                    );
                    tensor_buffers.insert(*output, output_buf);
                }

                Operation::SiLU { input, output } => {
                    let input_buf = tensor_buffers.get(input)?;
                    let output_buf = gpu_ctx.silu_encode(
                        encoder, input_buf,
                        1024, // Will be set dynamically
                    );
                    tensor_buffers.insert(*output, output_buf);
                }

                Operation::Mul { lhs, rhs, output } => {
                    let lhs_buf = tensor_buffers.get(lhs)?;
                    let rhs_buf = tensor_buffers.get(rhs)?;
                    let output_buf = gpu_ctx.mul_encode(
                        encoder, lhs_buf, rhs_buf,
                        1024, // Will be set dynamically
                    );
                    tensor_buffers.insert(*output, output_buf);
                }

                Operation::Add { lhs, rhs, output } => {
                    let lhs_buf = tensor_buffers.get(lhs)?;
                    let rhs_buf = tensor_buffers.get(rhs)?;
                    let output_buf = gpu_ctx.add_encode(
                        encoder, lhs_buf, rhs_buf,
                        1024, // Will be set dynamically
                    );
                    tensor_buffers.insert(*output, output_buf);
                }

                _ => {
                    // Handle other operations
                }
            }
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Get final output
        let final_node = dag.nodes.get(*dag.exit_points.last()?)?;
        if let Operation::Add { output, .. } = &final_node.op {
            tensor_buffers.get(output).cloned()
        } else {
            None
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn gemv_encode(
        &self,
        encoder: &ComputeCommandEncoder,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        out_size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<std::rc::Rc<metal::Buffer>> {
        // This will call the actual GEMV kernel
        // For now, return a placeholder
        let device = gpu_ctx.device();
        let output = std::rc::Rc::new(device.new_buffer(
            (out_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // TODO: Call actual GEMV kernel
        Some(output)
    }
}
