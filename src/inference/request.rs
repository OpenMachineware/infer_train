// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Request context for reentrant inference.

use std::collections::VecDeque;

use crate::backend::MemoryRegionId;

/// Request state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Request is currently being processed.
    Running,
    /// Request is waiting to be scheduled.
    Waiting,
    /// Request was preempted due to memory pressure.
    Preempted,
    /// Request has completed.
    Finished,
}

/// Request priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Low priority (background tasks).
    Low = 0,
    /// Normal priority (default).
    Normal = 1,
    /// High priority (interactive).
    High = 2,
    /// Real-time priority (urgent).
    RealTime = 3,
}

/// Request context for isolated inference.
///
/// Each request has its own:
/// - Input/output tokens
/// - KV cache (reference to external cache or owned cache)
/// - Inference parameters
/// - Scratch buffer (for intermediate results)
#[derive(Debug)]
pub struct RequestContext {
    /// Unique request identifier.
    pub request_id: u64,

    /// Current request state.
    pub state: RequestState,

    /// Input prompt tokens.
    pub prompt_tokens: Vec<u32>,

    /// Current position in prompt (for incremental prefill).
    pub prompt_pos: usize,

    /// Generated output tokens.
    pub generated_tokens: Vec<u32>,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Sampling temperature.
    pub temperature: f32,

    /// Request priority.
    pub priority: Priority,

    /// Memory region for this request's data.
    pub memory_region: MemoryRegionId,

    /// Scratch buffer ID (from RequestScratchPool).
    pub scratch_id: Option<usize>,
}

impl RequestContext {
    /// Create a new request context.
    pub fn new(
        request_id: u64,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        priority: Priority,
        memory_region: MemoryRegionId,
    ) -> Self {
        Self {
            request_id,
            state: RequestState::Waiting,
            prompt_tokens,
            prompt_pos: 0,
            generated_tokens: Vec::with_capacity(max_tokens),
            max_tokens,
            temperature,
            priority,
            memory_region,
            scratch_id: None,
        }
    }

    /// Get total sequence length (prompt + generated).
    pub fn seq_len(&self) -> usize {
        self.prompt_pos + self.generated_tokens.len()
    }

    /// Check if prompt has been fully processed.
    pub fn is_prefill_complete(&self) -> bool {
        self.prompt_pos >= self.prompt_tokens.len()
    }

    /// Check if generation is complete.
    pub fn is_complete(&self) -> bool {
        self.state == RequestState::Finished
            || self.generated_tokens.len() >= self.max_tokens
    }

    /// Get current token to process (from prompt or generated).
    pub fn current_token(&self) -> Option<u32> {
        if !self.is_prefill_complete() {
            self.prompt_tokens.get(self.prompt_pos).copied()
        } else {
            self.generated_tokens.last().copied()
        }
    }

    /// Advance prompt position by one.
    pub fn advance_prompt(&mut self) {
        if !self.is_prefill_complete() {
            self.prompt_pos += 1;
        }
    }

    /// Add a generated token.
    pub fn add_generated_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
    }

    /// Get all tokens (prompt + generated).
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut tokens = self.prompt_tokens.clone();
        tokens.extend(&self.generated_tokens);
        tokens
    }
}

/// Scratch buffer for request-level intermediate results.
///
/// Each request needs temporary buffers for:
/// - Hidden states
/// - Attention scores
/// - FFN intermediate results
#[derive(Debug)]
pub struct RequestScratch {
    /// Scratch buffer ID.
    pub id: usize,

    /// Hidden state buffer.
    pub hidden: Vec<f32>,

    /// Query buffer for attention.
    pub query: Vec<f32>,

    /// Key buffer for attention.
    pub key: Vec<f32>,

    /// Value buffer for attention.
    pub value: Vec<f32>,

    /// Attention output buffer.
    pub attn_out: Vec<f32>,

    /// FFN gate buffer.
    pub ffn_gate: Vec<f32>,

    /// FFN up buffer.
    pub ffn_up: Vec<f32>,

    /// FFN down buffer.
    pub ffn_down: Vec<f32>,

    /// Memory region for GPU buffers (if applicable).
    pub memory_region: MemoryRegionId,
}

impl RequestScratch {
    /// Create a new scratch buffer with given sizes.
    pub fn new(
        id: usize,
        hidden_size: usize,
        intermediate_size: usize,
        memory_region: MemoryRegionId,
    ) -> Self {
        Self {
            id,
            hidden: vec![0.0; hidden_size],
            query: vec![0.0; hidden_size],
            key: vec![0.0; hidden_size],
            value: vec![0.0; hidden_size],
            attn_out: vec![0.0; hidden_size],
            ffn_gate: vec![0.0; intermediate_size],
            ffn_up: vec![0.0; intermediate_size],
            ffn_down: vec![0.0; hidden_size],
            memory_region,
        }
    }

    /// Get total buffer size in bytes.
    pub fn total_size(&self) -> usize {
        (self.hidden.len()
            + self.query.len()
            + self.key.len()
            + self.value.len()
            + self.attn_out.len()
            + self.ffn_gate.len()
            + self.ffn_up.len()
            + self.ffn_down.len())
            * std::mem::size_of::<f32>()
    }
}

/// Pool of scratch buffers for request-level allocation.
///
/// Pre-allocates buffers to avoid per-request allocation overhead.
#[derive(Debug)]
pub struct RequestScratchPool {
    /// Available scratch buffers.
    free: Vec<RequestScratch>,

    /// Hidden size for each buffer.
    hidden_size: usize,

    /// Intermediate size for each buffer.
    intermediate_size: usize,

    /// Memory region for GPU buffers.
    memory_region: MemoryRegionId,
}

impl RequestScratchPool {
    /// Create a new scratch pool.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        memory_region: MemoryRegionId,
    ) -> Self {
        Self { free: Vec::new(), hidden_size, intermediate_size, memory_region }
    }

    /// Pre-allocate a number of scratch buffers.
    pub fn preallocate(&mut self, count: usize) {
        for i in 0..count {
            let scratch = RequestScratch::new(
                i,
                self.hidden_size,
                self.intermediate_size,
                self.memory_region,
            );
            self.free.push(scratch);
        }
    }

    /// Allocate a scratch buffer for a request.
    pub fn allocate(&mut self) -> Option<RequestScratch> {
        if let Some(mut scratch) = self.free.pop() {
            // Reset buffers to zero
            scratch.hidden.fill(0.0);
            scratch.query.fill(0.0);
            scratch.key.fill(0.0);
            scratch.value.fill(0.0);
            scratch.attn_out.fill(0.0);
            scratch.ffn_gate.fill(0.0);
            scratch.ffn_up.fill(0.0);
            scratch.ffn_down.fill(0.0);
            Some(scratch)
        } else {
            // Pool exhausted, allocate new buffer
            let id = self.free.len();
            Some(RequestScratch::new(
                id,
                self.hidden_size,
                self.intermediate_size,
                self.memory_region,
            ))
        }
    }

    /// Release a scratch buffer back to the pool.
    pub fn release(&mut self, scratch: RequestScratch) {
        self.free.push(scratch);
    }

    /// Get number of available buffers.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Get total memory used by pool in bytes.
    pub fn total_memory(&self) -> usize {
        self.free.iter().map(|s| s.total_size()).sum()
    }
}

/// Queue of pending requests.
#[derive(Debug)]
pub struct RequestQueue {
    /// Pending requests (sorted by priority).
    queue: VecDeque<RequestContext>,
}

impl RequestQueue {
    /// Create a new request queue.
    pub fn new() -> Self {
        Self { queue: VecDeque::new() }
    }

    /// Add a request to the queue.
    pub fn push(&mut self, request: RequestContext) {
        // Insert in priority order
        let pos = self
            .queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(self.queue.len());
        self.queue.insert(pos, request);
    }

    /// Get the next request.
    pub fn pop(&mut self) -> Option<RequestContext> {
        self.queue.pop_front()
    }

    /// Peek at the next request.
    pub fn peek(&self) -> Option<&RequestContext> {
        self.queue.front()
    }

    /// Get number of pending requests.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Remove a specific request by ID.
    pub fn remove(&mut self, request_id: u64) -> Option<RequestContext> {
        if let Some(pos) =
            self.queue.iter().position(|r| r.request_id == request_id)
        {
            self.queue.remove(pos)
        } else {
            None
        }
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}
