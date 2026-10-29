// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Request executor for multi-request inference.

use crate::cpu::CpuWorkBuffersPool;
use crate::inference::{
    InferenceContext, KVCache, KVCacheFormat, RequestContext,
};
use crate::ops::sampling::{SamplingStrategy, sample};

/// Request executor for managing multiple concurrent requests.
///
/// This executor provides:
/// - Request-level isolation (each request has its own state)
/// - Scratch buffer pooling (avoids per-request allocation)
/// - Multi-request scheduling (can interleave requests)
pub struct RequestExecutor {
    /// Inference context (shared model).
    ctx: InferenceContext,

    /// Work buffer pool for zero-allocation inference.
    buffer_pool: CpuWorkBuffersPool,
}

impl RequestExecutor {
    /// Create a new request executor.
    pub fn new(ctx: InferenceContext) -> Self {
        // Create buffer pool from model
        let mut buffer_pool = CpuWorkBuffersPool::new(ctx.model());

        // Pre-allocate 4 buffers for concurrent requests
        buffer_pool.preallocate(4);

        Self { ctx, buffer_pool }
    }

    /// Execute one step of a request (prefill or decode).
    ///
    /// Returns the generated token (if any) and updates request state.
    #[cfg(target_arch = "aarch64")]
    pub fn step(
        &mut self,
        request: &mut RequestContext,
        kv_cache: &mut KVCache,
        strategy: &SamplingStrategy,
    ) -> Option<u32> {
        // Allocate work buffer for this request
        let mut buffers = self.buffer_pool.allocate()?;

        // Determine current token(s) to process
        let tokens = if !request.is_prefill_complete() {
            // Prefill: process next prompt token
            vec![request.prompt_tokens[request.prompt_pos]]
        } else {
            // Decode: process last generated token
            vec![*request.generated_tokens.last()?]
        };

        // Run zero-allocation forward pass
        let logits =
            self.ctx.model().forward_pooled(&tokens, kv_cache, &mut buffers)?;

        // Release buffer back to pool
        self.buffer_pool.release(buffers);

        // Sample next token
        let next_token = sample(&logits, strategy) as u32;

        // Update request state
        if !request.is_prefill_complete() {
            request.advance_prompt();
        } else {
            request.add_generated_token(next_token);
        }

        Some(next_token)
    }

    /// Execute one step (fallback for non-aarch64).
    #[cfg(not(target_arch = "aarch64"))]
    pub fn step(
        &mut self,
        request: &mut RequestContext,
        kv_cache: &mut KVCache,
        strategy: &SamplingStrategy,
    ) -> Option<u32> {
        // Determine current token(s) to process
        let tokens = if !request.is_prefill_complete() {
            vec![request.prompt_tokens[request.prompt_pos]]
        } else {
            vec![*request.generated_tokens.last()?]
        };

        // Run standard forward pass
        let logits = self.ctx.forward(&tokens, kv_cache)?;

        // Sample next token
        let next_token = sample(&logits, strategy) as u32;

        // Update request state
        if !request.is_prefill_complete() {
            request.advance_prompt();
        } else {
            request.add_generated_token(next_token);
        }

        Some(next_token)
    }

    /// Run a request to completion.
    ///
    /// This processes all prompt tokens and generates max_tokens.
    pub fn run(
        &mut self,
        request: &mut RequestContext,
        strategy: &SamplingStrategy,
    ) -> Vec<u32> {
        // Create KV cache for this request
        let max_seq = request.prompt_tokens.len() + request.max_tokens;
        let mut kv_cache =
            KVCache::new(&self.ctx.config, max_seq, KVCacheFormat::F32);

        // Process prompt tokens (prefill)
        while !request.is_prefill_complete() {
            self.step(request, &mut kv_cache, strategy);
        }

        // Generate first token from prompt
        if let Some(first_token) = self.step(request, &mut kv_cache, strategy) {
            request.add_generated_token(first_token);
        }

        // Generate remaining tokens (decode)
        while !request.is_complete() {
            self.step(request, &mut kv_cache, strategy);
        }

        request.generated_tokens.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::Priority;

    #[test]
    fn test_request_executor_creation() {
        // This test requires a model file, so we skip it
        // In a real test, you would load a small model
    }

    #[test]
    fn test_buffer_pool() {
        // This test is now in tests/test_request_context.rs
        // CpuWorkBuffersPool requires a model, so we can't test it here
    }
}
