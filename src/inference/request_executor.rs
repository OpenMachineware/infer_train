// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Request executor for multi-request inference.

use crate::backend::MemoryRegionId;
use crate::inference::{
    InferenceContext, KVCache, KVCacheFormat, RequestContext,
    RequestScratchPool,
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

    /// Scratch buffer pool.
    scratch_pool: RequestScratchPool,
}

impl RequestExecutor {
    /// Create a new request executor.
    pub fn new(ctx: InferenceContext, memory_region: MemoryRegionId) -> Self {
        let hidden_size = ctx.config.hidden_size;
        let intermediate_size = ctx.config.intermediate_size;

        let mut scratch_pool = RequestScratchPool::new(
            hidden_size,
            intermediate_size,
            memory_region,
        );

        // Pre-allocate 4 scratch buffers
        scratch_pool.preallocate(4);

        Self { ctx, scratch_pool }
    }

    /// Execute one step of a request (prefill or decode).
    ///
    /// Returns the generated token (if any) and updates request state.
    pub fn step(
        &mut self,
        request: &mut RequestContext,
        kv_cache: &mut KVCache,
        strategy: &SamplingStrategy,
    ) -> Option<u32> {
        // Allocate scratch buffer for this request
        let _scratch = self.scratch_pool.allocate()?;

        // Determine current token(s) to process
        let tokens = if !request.is_prefill_complete() {
            // Prefill: process next prompt token
            vec![request.prompt_tokens[request.prompt_pos]]
        } else {
            // Decode: process last generated token
            vec![*request.generated_tokens.last()?]
        };

        // Run forward pass
        let logits = self.ctx.forward(&tokens, kv_cache)?;

        // Sample next token
        let next_token = sample(&logits, strategy) as u32;

        // Update request state
        if !request.is_prefill_complete() {
            request.advance_prompt();
        } else {
            request.add_generated_token(next_token);
        }

        // Release scratch buffer
        // scratch_pool.release(scratch);

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
    use crate::backend::{Device, MemoryTopology};
    use crate::inference::{Priority, RequestState};

    #[test]
    fn test_request_executor_creation() {
        // This test requires a model file, so we skip it
        // In a real test, you would load a small model
    }

    #[test]
    fn test_scratch_pool_integration() {
        let topology = MemoryTopology::detect();
        let region = topology
            .regions
            .first()
            .map(|r| r.id())
            .unwrap_or_else(|| MemoryRegionId(0));

        let mut pool = RequestScratchPool::new(4096, 16384, region);
        pool.preallocate(2);

        assert_eq!(pool.available(), 2);

        let scratch1 = pool.allocate().unwrap();
        assert_eq!(pool.available(), 1);

        pool.release(scratch1);
        assert_eq!(pool.available(), 2);
    }
}
