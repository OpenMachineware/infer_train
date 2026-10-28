// SPDX-License-Identifier: AGPL-3.0-or-later
//
// CPU Memory Pool - Pre-allocated buffers for zero-allocation inference
//
// This module provides a memory pool that pre-allocates all intermediate buffers
// needed for CPU inference. This eliminates the per-forward allocation overhead
// (448 allocations per forward pass) that was causing the 22x performance gap
// compared to llama.cpp.

/// Buffer identifiers for tracking pre-allocated memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferId {
    // Hidden states (reused across layers)
    Hidden,    // Main hidden state
    HiddenAlt, // Alternate hidden state (for residual computation)

    // Norm outputs (reused)
    AttnNorm, // Attention layer norm output
    FfnNorm,  // FFN layer norm output

    // QKV projections (reused within layer)
    Query, // Query projection
    Key,   // Key projection
    Value, // Value projection

    // RoPE outputs (reused within layer)
    QueryRoPE, // Query after RoPE
    KeyRoPE,   // Key after RoPE

    // Attention outputs (reused)
    AttentionOut, // Attention output
    AttnProj,     // Attention projection output

    // FFN intermediate (reused within FFN)
    FfnGate,   // FFN gate projection
    FfnUp,     // FFN up projection
    FfnHidden, // FFN hidden state (gate * silu(up))
    FfnOut,    // FFN output

    // Final output
    Logits, // Final logits (vocab_size)

    // Attention scores (variable size, seq_len)
    AttentionScores,
}

/// Pre-allocated buffer with metadata
struct Buffer {
    data: Vec<f32>,
    size: usize,
    in_use: bool,
}

impl Buffer {
    fn new(size: usize) -> Self {
        Buffer { data: vec![0.0f32; size], size, in_use: false }
    }
}

/// CPU Memory Pool
///
/// Pre-allocates all buffers needed for a forward pass and reuses them
/// across multiple forward passes. This eliminates allocation overhead.
///
/// Usage:
/// 1. Call `get_mut_vec(id)` to get a mutable buffer
/// 2. Write operator output to the buffer
/// 3. Pass buffer slice to next operator
/// 4. Call `reset()` after forward pass
#[allow(dead_code)]
pub struct CpuMemoryPool {
    buffers: Vec<Buffer>,

    // Configuration-dependent sizes
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    q_size: usize,
    k_size: usize,
    v_size: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
}

impl CpuMemoryPool {
    /// Create a new memory pool with pre-allocated buffers
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        q_size: usize,
        k_size: usize,
        v_size: usize,
        max_seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let v_head_dim = v_size / n_kv_heads;

        // Pre-allocate buffers based on size categories
        let mut buffers = Vec::new();

        // Hidden states (size = hidden_size)
        for _ in 0..2 {
            buffers.push(Buffer::new(hidden_size));
        }

        // Norm outputs (size = hidden_size)
        for _ in 0..2 {
            buffers.push(Buffer::new(hidden_size));
        }

        // Q/K/V projections (varying sizes)
        buffers.push(Buffer::new(q_size)); // Query
        buffers.push(Buffer::new(k_size)); // Key
        buffers.push(Buffer::new(v_size)); // Value

        // RoPE outputs
        buffers.push(Buffer::new(q_size)); // QueryRoPE
        buffers.push(Buffer::new(k_size)); // KeyRoPE

        // Attention outputs
        buffers.push(Buffer::new(n_heads * v_head_dim)); // AttentionOut
        buffers.push(Buffer::new(hidden_size)); // AttnProj

        // FFN intermediate
        buffers.push(Buffer::new(intermediate_size)); // FfnGate
        buffers.push(Buffer::new(intermediate_size)); // FfnUp
        buffers.push(Buffer::new(intermediate_size)); // FfnHidden
        buffers.push(Buffer::new(hidden_size)); // FfnOut

        // Final logits
        buffers.push(Buffer::new(vocab_size));

        // Attention scores (max_seq_len)
        buffers.push(Buffer::new(max_seq_len));

        CpuMemoryPool {
            buffers,
            hidden_size,
            intermediate_size,
            vocab_size,
            q_size,
            k_size,
            v_size,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            v_head_dim,
        }
    }

    /// Get a buffer as a mutable Vec (borrows the whole pool)
    /// This is the primary interface for operators
    pub fn get_mut_vec(&mut self, id: BufferId) -> Option<&mut Vec<f32>> {
        let idx = self.buffer_index(id)?;
        let buffer = &mut self.buffers[idx];
        buffer.in_use = true;
        Some(&mut buffer.data)
    }

    /// Get a buffer as an immutable slice
    pub fn get_slice(&self, id: BufferId) -> Option<&[f32]> {
        let idx = self.buffer_index(id)?;
        Some(&self.buffers[idx].data[..])
    }

    /// Write data to a buffer
    pub fn write(&mut self, id: BufferId, data: &[f32]) -> bool {
        let idx = match self.buffer_index(id) {
            Some(i) => i,
            None => return false,
        };

        let buffer = &mut self.buffers[idx];

        if data.len() <= buffer.size {
            buffer.data[..data.len()].copy_from_slice(data);
            buffer.in_use = true;
            true
        } else {
            false
        }
    }

    /// Read data from a buffer (returns copy for safety)
    pub fn read(&self, id: BufferId) -> Option<Vec<f32>> {
        let idx = self.buffer_index(id)?;
        Some(self.buffers[idx].data.clone())
    }

    /// Reset all buffers (call at end of forward pass)
    pub fn reset(&mut self) {
        for buffer in self.buffers.iter_mut() {
            buffer.in_use = false;
            // Clear data
            for elem in buffer.data.iter_mut() {
                *elem = 0.0;
            }
        }
    }

    /// Get buffer size for a given ID
    pub fn buffer_size(&self, id: BufferId) -> usize {
        match id {
            BufferId::Hidden | BufferId::HiddenAlt => self.hidden_size,
            BufferId::AttnNorm | BufferId::FfnNorm => self.hidden_size,
            BufferId::Query => self.q_size,
            BufferId::Key => self.k_size,
            BufferId::Value => self.v_size,
            BufferId::QueryRoPE => self.q_size,
            BufferId::KeyRoPE => self.k_size,
            BufferId::AttentionOut => self.n_heads * self.v_head_dim,
            BufferId::AttnProj => self.hidden_size,
            BufferId::FfnGate | BufferId::FfnUp | BufferId::FfnHidden => {
                self.intermediate_size
            }
            BufferId::FfnOut => self.hidden_size,
            BufferId::Logits => self.vocab_size,
            BufferId::AttentionScores => self.max_seq_len,
        }
    }

    /// Map BufferId to index in buffers array
    fn buffer_index(&self, id: BufferId) -> Option<usize> {
        Some(match id {
            BufferId::Hidden => 0,
            BufferId::HiddenAlt => 1,
            BufferId::AttnNorm => 2,
            BufferId::FfnNorm => 3,
            BufferId::Query => 4,
            BufferId::Key => 5,
            BufferId::Value => 6,
            BufferId::QueryRoPE => 7,
            BufferId::KeyRoPE => 8,
            BufferId::AttentionOut => 9,
            BufferId::AttnProj => 10,
            BufferId::FfnGate => 11,
            BufferId::FfnUp => 12,
            BufferId::FfnHidden => 13,
            BufferId::FfnOut => 14,
            BufferId::Logits => 15,
            BufferId::AttentionScores => 16,
        })
    }

    /// Get total memory allocated
    pub fn total_memory(&self) -> usize {
        self.buffers.iter().map(|b| b.size * std::mem::size_of::<f32>()).sum()
    }

    /// Get the number of buffers
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}
