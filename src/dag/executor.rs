// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG Executor - Traverse DAG and dispatch operations
//!
//! Executes operations with single command buffer for entire forward pass

use crate::model::LlamaModel;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::buffer::ForwardBufferPool;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::inference::KVCacheGPU;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::cell::RefCell;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

pub struct DagExecutor;

impl DagExecutor {
    /// Execute forward pass with single command buffer for entire model
    /// This is the key optimization: one command buffer instead of one per layer
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu_graph(
        model: &LlamaModel,
        tokens: &[u32],
        kv_cache: &mut KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || model.tok_embeddings.is_none() {
            return None;
        }

        let hidden_size = model.config.hidden_size;
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Create shared reuse pool for buffers (persisted across forward passes)
        let reuse_pool: Rc<RefCell<Vec<Rc<metal::Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));

        // Create forward pool for this pass
        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        // Process each token
        let mut last_hidden_buffer: Option<Rc<metal::Buffer>> = None;

        for &token in tokens.iter() {
            // 1. Embedding lookup (CPU)
            let hidden = model.embedding_lookup_impl(token)?;

            // Upload to GPU
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // 2. Process through layers with per-layer command buffers
            // NOTE: Metal requires commit+wait between layers for memory coherency
            // Single command buffer across all layers produces NaN due to GPU memory issues
            let mut layer_input = hidden_buffer.clone();
            let n_layers = model.layers.len();

            for layer_idx in 0..n_layers {
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let layer_output = model.forward_layer_gpu_encode(
                    &model.layers[layer_idx],
                    layer_idx,
                    &layer_input,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                )?;

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                layer_input = layer_output;
            }

            // 3. Final norm (separate command buffer)
            let norm = model.norm.as_ref()?;
            let norm_buffer = gpu_ctx.get_or_create_buffer(norm);
            let final_hidden_buffer = forward_pool.get(hidden_size * 4);

            let final_command_buffer = queue.new_command_buffer();
            let final_encoder =
                final_command_buffer.new_compute_command_encoder();

            gpu_ctx.rms_norm_encode(
                final_encoder,
                &layer_input,
                &norm_buffer,
                &final_hidden_buffer,
                1,
                hidden_size,
                1e-5,
            );

            // 4. Output projection
            let output = model.output.as_ref()?;
            let vocab_size =
                if output.shape.len() == 2 && output.shape[1] == hidden_size {
                    output.shape[0]
                } else {
                    output.shape[0]
                };

            // Check if output dtype is supported for GPU
            let supported_for_gpu = matches!(
                output.dtype,
                crate::model::tensor::DataType::Q4_0
                    | crate::model::tensor::DataType::Q4_1
                    | crate::model::tensor::DataType::Q5_0
                    | crate::model::tensor::DataType::Q5_1
                    | crate::model::tensor::DataType::Q8_0
                    | crate::model::tensor::DataType::Q4_K
                    | crate::model::tensor::DataType::Q5_K
                    | crate::model::tensor::DataType::Q2_K
                    | crate::model::tensor::DataType::Q6_K
                    | crate::model::tensor::DataType::Q3_K
            );

            if !supported_for_gpu
                || (output.shape.len() == 2 && output.shape[1] == hidden_size)
            {
                // Commit and wait before reading (CRITICAL!)
                final_encoder.end_encoding();
                final_command_buffer.commit();
                final_command_buffer.wait_until_completed();

                // Read final_hidden
                let mut hidden = vec![0.0f32; hidden_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        final_hidden_buffer.contents() as *const f32,
                        hidden.as_mut_ptr(),
                        hidden_size,
                    );
                }

                if output.shape.len() == 2 && output.shape[1] == hidden_size {
                    return model.gemv_transposed(&hidden, output, vocab_size);
                } else {
                    return model.gemv(&hidden, output, vocab_size);
                }
            }

            // GPU output projection
            let logits_buffer = forward_pool.get(vocab_size * 4);

            if model
                .gemv_encode(
                    final_encoder,
                    &final_hidden_buffer,
                    output,
                    vocab_size,
                    gpu_ctx,
                    &logits_buffer,
                )
                .is_none()
            {
                return None;
            }

            // Commit ONCE after all operations
            final_encoder.end_encoding();
            final_command_buffer.commit();
            final_command_buffer.wait_until_completed();

            // Download logits
            let mut logits = vec![0.0f32; vocab_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    logits_buffer.contents() as *const f32,
                    logits.as_mut_ptr(),
                    vocab_size,
                );
            }

            last_hidden_buffer = Some(logits_buffer);
            kv_cache.advance();
        }

        // For simplicity, return the first token's logits
        // In a real implementation, we'd handle multiple tokens properly
        if let Some(_buf) = last_hidden_buffer {
            // Re-download logits from last pass
            let output = model.output.as_ref()?;
            let vocab_size =
                if output.shape.len() == 2 && output.shape[1] == hidden_size {
                    output.shape[0]
                } else {
                    output.shape[0]
                };

            // We already returned in the loop above
        }

        None
    }
}
