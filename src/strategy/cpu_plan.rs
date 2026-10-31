// SPDX-License-Identifier: AGPL-3.0-or-later
//
// CPU execution plan with zero-allocation

use crate::cpu::CpuWorkBuffers;
use crate::inference::KVCache;
use crate::model::llama::LlamaModel;

/// CPU execution plan
///
/// Pre-compiled execution plan for CPU inference.
/// Encapsulates all operators and buffer management.
pub struct CpuExecutionPlan {
    /// Model reference
    model: std::sync::Arc<LlamaModel>,

    /// Pre-allocated work buffers
    buffers: CpuWorkBuffers,
}

impl CpuExecutionPlan {
    /// Create a new CPU execution plan from a model
    pub fn new(model: std::sync::Arc<LlamaModel>) -> Self {
        let config = &model.config;

        // Get qkv_out_size from first layer
        let qkv_out_size = model
            .layers
            .first()
            .and_then(|l| l.attn_qkv.as_ref())
            .map(|t| t.shape[1])
            .unwrap_or(0);

        let buffers = CpuWorkBuffers::new(
            config.hidden_size,
            config.intermediate_size,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.key_length,
            config.value_length,
            config.vocab_size,
            qkv_out_size,
            2048, // max_seq_len
            model.layers.len(),
        );

        Self { model, buffers }
    }

    /// Execute forward pass for tokens
    ///
    /// Automatically uses decode path for single token, prefill for multiple.
    #[cfg(target_arch = "aarch64")]
    pub fn forward(
        &mut self,
        tokens: &[u32],
        kv_cache: &mut KVCache,
    ) -> Option<Vec<f32>> {
        self.model.forward_pooled(tokens, kv_cache, &mut self.buffers)
    }

    /// Get the model reference
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }
}

impl std::fmt::Debug for CpuExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutionPlan")
            .field("hidden_size", &self.model.config.hidden_size)
            .field("intermediate_size", &self.model.config.intermediate_size)
            .field("n_layers", &self.model.layers.len())
            .finish()
    }
}
