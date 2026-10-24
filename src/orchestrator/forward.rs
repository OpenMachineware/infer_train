// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Forward pass orchestrator (Layer 3).
//!
//! Coordinates forward pass execution across GPU and CPU backends.
//! Handles architecture detection, path selection, and KV cache lifecycle.

use crate::buffer::ComputeDevice;
use crate::inference::{KVCache, KVCacheBase, KVCacheFormat};
use crate::model::llama::LlamaModel;
use crate::orchestrator::architecture::Architecture;
use std::sync::Arc;

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

    /// Detected architecture
    architecture: Architecture,

    /// KV cache
    kv_cache: KVCache,

    /// Preferred device
    preferred_device: ComputeDevice,
}

impl ForwardOrchestrator {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn new(model: Arc<LlamaModel>) -> Self {
        let architecture = Architecture::detect(&model);

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

        Self { model, architecture, kv_cache, preferred_device }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    pub fn new(model: Arc<LlamaModel>) -> Self {
        let architecture = Architecture::detect(&model);

        // Pre-allocate KV cache
        let kv_cache = KVCache::new(
            &model.config,
            model.config.max_position_embeddings,
            KVCacheFormat::F16,
        );

        Self {
            model,
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

    /// Forward pass for tokens (device-aware)
    ///
    /// This is the main entry point for inference.
    /// Automatically selects GPU or CPU path based on architecture.
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCache,
    ) -> Option<Vec<f32>> {
        if self.preferred_device == ComputeDevice::LocalGpu
            && self.architecture.is_gpu_supported()
        {
            // Use optimized GPU path
            // Note: This requires KVCacheGPU, which is handled separately
            // For now, fall back to CPU
            self.model.forward(tokens, kv_cache)
        } else {
            // CPU path
            self.model.forward(tokens, kv_cache)
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    pub fn forward(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCache,
    ) -> Option<Vec<f32>> {
        self.model.forward(tokens, kv_cache)
    }

    /// Forward pass with GPU KV cache (zero-copy)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if self.architecture.is_gpu_supported() {
            // Use optimized batched GPU path
            self.model.forward_gpu_native_batched(tokens, kv_cache, gpu_ctx)
        } else {
            // Fall back to CPU (requires converting KV cache)
            None
        }
    }

    /// Forward pass for a single token (CPU, uses internal KV cache)
    pub fn forward_token(&mut self, token: u32) -> Option<Vec<f32>> {
        self.model.forward(&[token], &mut self.kv_cache)
    }

    /// Forward pass for prefill (uses internal KV cache)
    pub fn forward_prefill(&mut self, tokens: &[u32]) -> Option<Vec<f32>> {
        self.model.forward(tokens, &mut self.kv_cache)
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
