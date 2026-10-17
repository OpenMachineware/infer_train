// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Forward pass orchestrator (Layer 3).
//!
//! Coordinates forward pass execution across GPU and CPU backends.
//! Handles architecture detection, path selection, and KV cache lifecycle.

use crate::buffer::{BufferManager, ComputeDevice};
use crate::executor::OperationExecutor;
use crate::inference::{KVCache, KVCacheBase, KVCacheFormat};
use crate::model::llama::LlamaModel;
use crate::orchestrator::architecture::Architecture;
use std::sync::{Arc, Mutex};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::quant::vec_dot::gpu::MetalContext;

/// Forward pass orchestrator
///
/// Coordinates:
/// 1. Architecture detection
/// 2. Operation scheduling
/// 3. GPU-first with CPU fallback
/// 4. KV cache lifecycle
pub struct ForwardOrchestrator {
    /// Model reference
    model: Arc<LlamaModel>,

    /// Buffer manager
    buffer_mgr: Arc<Mutex<BufferManager>>,

    /// Operation executor
    executor: OperationExecutor,

    /// Detected architecture
    architecture: Architecture,

    /// KV cache
    kv_cache: KVCache,

    /// Preferred device
    preferred_device: ComputeDevice,
}

impl ForwardOrchestrator {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn new(
        model: Arc<LlamaModel>,
        buffer_mgr: Arc<Mutex<BufferManager>>,
        gpu_ctx: Arc<Mutex<MetalContext>>,
    ) -> Self {
        let architecture = Architecture::detect(&model);
        let executor = OperationExecutor::new(buffer_mgr.clone(), gpu_ctx);

        // Pre-allocate KV cache
        let kv_cache = KVCache::new(
            &model.config,
            model.config.max_position_embeddings,
            KVCacheFormat::F16,
        );

        // GPU-first if architecture is supported
        let preferred_device = if architecture.is_gpu_supported() {
            ComputeDevice::LocalGpu
        } else {
            ComputeDevice::Cpu
        };

        Self {
            model,
            buffer_mgr,
            executor,
            architecture,
            kv_cache,
            preferred_device,
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    pub fn new(
        model: Arc<LlamaModel>,
        buffer_mgr: Arc<Mutex<BufferManager>>,
    ) -> Self {
        let architecture = Architecture::detect(&model);
        let executor = OperationExecutor::new(buffer_mgr.clone());

        // Pre-allocate KV cache
        let kv_cache = KVCache::new(
            &model.config,
            model.config.max_position_embeddings,
            KVCacheFormat::F16,
        );

        Self {
            model,
            buffer_mgr,
            executor,
            architecture,
            kv_cache,
            preferred_device: ComputeDevice::Cpu,
        }
    }

    /// Get detected architecture
    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }

    /// Get KV cache reference
    pub fn kv_cache(&self) -> &KVCache {
        &self.kv_cache
    }

    /// Get mutable KV cache reference
    pub fn kv_cache_mut(&mut self) -> &mut KVCache {
        &mut self.kv_cache
    }

    /// Forward pass for tokens (CPU version)
    ///
    /// This is the main entry point for inference.
    /// Uses the model's existing forward implementation.
    pub fn forward(&mut self, tokens: &[u32]) -> Option<Vec<f32>> {
        self.model.forward(tokens, &mut self.kv_cache)
    }

    /// Forward pass for a single token
    pub fn forward_token(&mut self, token: u32) -> Option<Vec<f32>> {
        self.forward(&[token])
    }

    /// Forward pass for prefill (processing prompt tokens)
    pub fn forward_prefill(&mut self, tokens: &[u32]) -> Option<Vec<f32>> {
        self.forward(tokens)
    }

    /// Reset for new sequence
    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }

    /// Get current position
    pub fn current_pos(&self) -> usize {
        self.kv_cache.seq_len()
    }

    /// Get preferred device
    pub fn preferred_device(&self) -> ComputeDevice {
        self.preferred_device
    }
}
