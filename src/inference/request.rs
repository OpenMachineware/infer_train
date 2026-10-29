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

    // === Hidden states ===
    /// Hidden state buffer.
    pub hidden: Vec<f32>,

    /// Alternate hidden state buffer.
    pub hidden_alt: Vec<f32>,

    /// Normed hidden state.
    pub normed: Vec<f32>,

    // === Attention buffers ===
    /// Query buffer for attention.
    pub query: Vec<f32>,

    /// Key buffer for attention.
    pub key: Vec<f32>,

    /// Value buffer for attention.
    pub value: Vec<f32>,

    /// Attention output buffer.
    pub attn_out: Vec<f32>,

    /// Attention scores buffer.
    pub attn_scores: Vec<f32>,

    // === FFN buffers ===
    /// FFN gate buffer.
    pub ffn_gate: Vec<f32>,

    /// FFN up buffer.
    pub ffn_up: Vec<f32>,

    /// FFN down / hidden buffer.
    pub ffn_down: Vec<f32>,

    // === Output buffers ===
    /// Logits buffer.
    pub logits: Vec<f32>,

    // === Work buffers for quantization ===
    /// Q8_K input buffer.
    pub q8k_input: Vec<u8>,

    /// Padded input buffer.
    pub x_padded: Vec<f32>,

    // === RoPE buffer ===
    /// RoPE output buffer.
    pub rope_buffer: Vec<f32>,

    // === QKV buffer for fused QKV ===
    /// Fused QKV output buffer.
    pub qkv_buffer: Vec<f32>,

    /// Memory region for GPU buffers (if applicable).
    pub memory_region: MemoryRegionId,
}

impl RequestScratch {
    /// Create a new scratch buffer with given sizes.
    pub fn new(
        id: usize,
        hidden_size: usize,
        intermediate_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_length: usize,
        value_length: usize,
        vocab_size: usize,
        max_seq_len: usize,
        memory_region: MemoryRegionId,
    ) -> Self {
        // Calculate buffer sizes
        let q_size = n_heads * head_dim;
        let k_size = if key_length > 0 {
            n_kv_heads * key_length
        } else {
            n_kv_heads * head_dim
        };
        let v_size = if value_length > 0 {
            n_kv_heads * value_length
        } else {
            n_kv_heads * head_dim
        };
        let kv_size = k_size.max(v_size);
        let v_head_dim = if value_length > 0 { value_length } else { head_dim };
        let attn_out_size = n_heads * v_head_dim;

        Self {
            id,
            // Hidden states
            hidden: vec![0.0; hidden_size],
            hidden_alt: vec![0.0; hidden_size],
            normed: vec![0.0; hidden_size],
            // Attention
            query: vec![0.0; q_size],
            key: vec![0.0; kv_size],
            value: vec![0.0; kv_size],
            attn_out: vec![0.0; attn_out_size],
            attn_scores: vec![0.0; max_seq_len * n_heads],
            // FFN
            ffn_gate: vec![0.0; intermediate_size],
            ffn_up: vec![0.0; intermediate_size],
            ffn_down: vec![0.0; hidden_size],
            // Output
            logits: vec![0.0; vocab_size],
            // Work buffers
            q8k_input: vec![0; 65536],
            x_padded: vec![0.0; 65536],
            // RoPE
            rope_buffer: vec![0.0; q_size.max(kv_size)],
            // Fused QKV
            qkv_buffer: vec![0.0; q_size + k_size + v_size],
            memory_region,
        }
    }

    /// Get total buffer size in bytes.
    pub fn total_size(&self) -> usize {
        (self.hidden.len()
            + self.hidden_alt.len()
            + self.normed.len()
            + self.query.len()
            + self.key.len()
            + self.value.len()
            + self.attn_out.len()
            + self.attn_scores.len()
            + self.ffn_gate.len()
            + self.ffn_up.len()
            + self.ffn_down.len()
            + self.logits.len()
            + self.x_padded.len()
            + self.rope_buffer.len()
            + self.qkv_buffer.len())
            * std::mem::size_of::<f32>()
            + self.q8k_input.len()
    }
}

/// Pool of scratch buffers for request-level allocation.
///
/// Pre-allocates buffers to avoid per-request allocation overhead.
#[derive(Debug)]
pub struct RequestScratchPool {
    /// Available scratch buffers.
    free: Vec<RequestScratch>,

    /// Buffer configuration.
    hidden_size: usize,
    intermediate_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    key_length: usize,
    value_length: usize,
    vocab_size: usize,
    max_seq_len: usize,

    /// Memory region for GPU buffers.
    memory_region: MemoryRegionId,
}

impl RequestScratchPool {
    /// Create a new scratch pool with model configuration.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_length: usize,
        value_length: usize,
        vocab_size: usize,
        max_seq_len: usize,
        memory_region: MemoryRegionId,
    ) -> Self {
        Self {
            free: Vec::new(),
            hidden_size,
            intermediate_size,
            n_heads,
            n_kv_heads,
            head_dim,
            key_length,
            value_length,
            vocab_size,
            max_seq_len,
            memory_region,
        }
    }

    /// Pre-allocate a number of scratch buffers.
    pub fn preallocate(&mut self, count: usize) {
        for i in 0..count {
            let scratch = RequestScratch::new(
                i,
                self.hidden_size,
                self.intermediate_size,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.key_length,
                self.value_length,
                self.vocab_size,
                self.max_seq_len,
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
            scratch.hidden_alt.fill(0.0);
            scratch.normed.fill(0.0);
            scratch.query.fill(0.0);
            scratch.key.fill(0.0);
            scratch.value.fill(0.0);
            scratch.attn_out.fill(0.0);
            scratch.attn_scores.fill(0.0);
            scratch.ffn_gate.fill(0.0);
            scratch.ffn_up.fill(0.0);
            scratch.ffn_down.fill(0.0);
            scratch.logits.fill(0.0);
            scratch.rope_buffer.fill(0.0);
            scratch.qkv_buffer.fill(0.0);
            Some(scratch)
        } else {
            // Pool exhausted, allocate new buffer
            let id = self.free.len();
            Some(RequestScratch::new(
                id,
                self.hidden_size,
                self.intermediate_size,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.key_length,
                self.value_length,
                self.vocab_size,
                self.max_seq_len,
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
