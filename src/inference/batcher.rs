// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Continuous batching for multi-request inference.

use std::collections::{HashMap, VecDeque};

use crate::cpu::CpuWorkBuffersPool;
use crate::inference::{
    InferenceContext, KVCache, KVCacheFormat, RequestContext, RequestState,
};
use crate::ops::sampling::{SamplingStrategy, sample};

/// Continuous batcher for managing multiple concurrent requests.
///
/// Simplified for local inference:
/// - Max 4-8 concurrent requests
/// - Simple preemption (release KV cache, put back in queue)
/// - Batch inference for throughput
pub struct ContinuousBatcher {
    /// Inference context (shared model).
    ctx: InferenceContext,

    /// All requests by ID.
    requests: HashMap<u64, RequestContext>,

    /// KV caches by request ID.
    kv_caches: HashMap<u64, KVCache>,

    /// Running request IDs (currently being processed).
    running: Vec<u64>,

    /// Waiting queue (sorted by priority).
    waiting: VecDeque<u64>,

    /// Completed request IDs (to be collected).
    completed: Vec<u64>,

    /// Work buffer pool.
    buffer_pool: CpuWorkBuffersPool,

    /// Configuration.
    max_batch_size: usize,
}

impl ContinuousBatcher {
    /// Create a new continuous batcher.
    pub fn new(ctx: InferenceContext, max_batch_size: usize) -> Self {
        let buffer_pool = CpuWorkBuffersPool::new(ctx.model());
        let buffer_pool = {
            let mut pool = buffer_pool;
            pool.preallocate(max_batch_size);
            pool
        };

        Self {
            ctx,
            requests: HashMap::new(),
            kv_caches: HashMap::new(),
            running: Vec::new(),
            waiting: VecDeque::new(),
            completed: Vec::new(),
            buffer_pool,
            max_batch_size,
        }
    }

    /// Add a new request to the batcher.
    pub fn add_request(&mut self, request: RequestContext) {
        let id = request.request_id;
        self.requests.insert(id, request);
        self.waiting.push_back(id);
    }

    /// Remove a completed request.
    pub fn remove_request(&mut self, id: u64) -> Option<RequestContext> {
        self.kv_caches.remove(&id);
        self.running.retain(|&x| x != id);
        self.requests.remove(&id)
    }

    /// Get request by ID.
    pub fn get_request(&self, id: u64) -> Option<&RequestContext> {
        self.requests.get(&id)
    }

    /// Get number of active requests (running + waiting).
    pub fn active_count(&self) -> usize {
        self.running.len() + self.waiting.len()
    }

    /// Get number of running requests.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Schedule waiting requests to running.
    fn schedule(&mut self) {
        while self.running.len() < self.max_batch_size {
            if let Some(id) = self.waiting.pop_front() {
                let request = self.requests.get_mut(&id).unwrap();
                request.state = RequestState::Running;
                self.running.push(id);

                // Create KV cache for this request if needed
                if !self.kv_caches.contains_key(&id) {
                    let max_seq =
                        request.prompt_tokens.len() + request.max_tokens;
                    let kv_cache = KVCache::new(
                        &self.ctx.config,
                        max_seq,
                        KVCacheFormat::F32,
                    );
                    self.kv_caches.insert(id, kv_cache);
                }
            } else {
                break;
            }
        }
    }

    /// Execute one step for all running requests.
    ///
    /// Returns generated tokens by request ID.
    #[cfg(target_arch = "aarch64")]
    pub fn step(&mut self, strategy: &SamplingStrategy) -> Vec<(u64, u32)> {
        // Schedule new requests
        self.schedule();

        // Process each running request
        let mut results = Vec::with_capacity(self.running.len());
        let mut to_complete = Vec::new();

        for &id in &self.running.clone() {
            let request = self.requests.get_mut(&id).unwrap();
            let kv_cache = self.kv_caches.get_mut(&id).unwrap();

            // Allocate work buffer
            let mut buffers = self.buffer_pool.allocate().unwrap();

            // Determine current token(s)
            let tokens = if !request.is_prefill_complete() {
                vec![request.prompt_tokens[request.prompt_pos]]
            } else {
                vec![
                    *request
                        .generated_tokens
                        .last()
                        .unwrap_or(&self.ctx.config.bos_token_id),
                ]
            };

            // Run forward
            let logits = self.ctx.model().forward_pooled(
                &tokens,
                kv_cache,
                &mut buffers,
            );

            // Release buffer
            self.buffer_pool.release(buffers);

            if let Some(logits) = logits {
                // Sample
                let token = sample(&logits, strategy) as u32;

                // Update state
                if !request.is_prefill_complete() {
                    request.advance_prompt();
                } else {
                    request.add_generated_token(token);

                    // Check completion
                    if request.is_complete() {
                        request.state = RequestState::Finished;
                        to_complete.push(id);
                    }
                }

                results.push((id, token));
            }
        }

        // Move completed requests
        for id in to_complete {
            self.running.retain(|&x| x != id);
            self.completed.push(id);
        }

        results
    }

    /// Execute one step (fallback for non-aarch64).
    #[cfg(not(target_arch = "aarch64"))]
    pub fn step(&mut self, strategy: &SamplingStrategy) -> Vec<(u64, u32)> {
        self.schedule();

        let mut results = Vec::with_capacity(self.running.len());
        let mut to_complete = Vec::new();

        for &id in &self.running.clone() {
            let request = self.requests.get_mut(&id).unwrap();
            let kv_cache = self.kv_caches.get_mut(&id).unwrap();

            let tokens = if !request.is_prefill_complete() {
                vec![request.prompt_tokens[request.prompt_pos]]
            } else {
                vec![
                    *request
                        .generated_tokens
                        .last()
                        .unwrap_or(&self.ctx.config.bos_token_id),
                ]
            };

            let logits = self.ctx.forward(&tokens, kv_cache);

            if let Some(logits) = logits {
                let token = sample(&logits, strategy) as u32;

                if !request.is_prefill_complete() {
                    request.advance_prompt();
                } else {
                    request.add_generated_token(token);
                    if request.is_complete() {
                        request.state = RequestState::Finished;
                        to_complete.push(id);
                    }
                }

                results.push((id, token));
            }
        }

        for id in to_complete {
            self.running.retain(|&x| x != id);
            self.completed.push(id);
        }

        results
    }

    /// Run until all requests complete.
    pub fn run_all(&mut self, strategy: &SamplingStrategy) {
        while !self.waiting.is_empty() || !self.running.is_empty() {
            self.step(strategy);
        }
    }

    /// Get completed request IDs.
    pub fn completed_ids(&self) -> &[u64] {
        &self.completed
    }

    /// Clear completed requests.
    pub fn clear_completed(&mut self) {
        for &id in &self.completed {
            self.kv_caches.remove(&id);
            self.requests.remove(&id);
        }
        self.completed.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batcher_creation() {
        // This test requires a model file
    }
}
