// Mamba/SSM (State Space Model) layer definition
// Used by models like Qwen3.8-27B, Jamba, etc.

use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;

/// Mamba/SSM layer with state space model
pub struct MambaLayer {
    pub layer_idx: usize,

    // Attention (may not exist in pure Mamba layers)
    pub attention_norm: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub wo: Option<Tensor>,

    // SSM (State Space Model)
    pub ssm_norm: Option<Tensor>,
    pub ssm_in: Option<Tensor>,     // Input projection
    pub ssm_out: Option<Tensor>,    // Output projection
    pub ssm_conv1d: Option<Tensor>, // 1D convolution
    pub ssm_a: Option<Tensor>,      // A parameter
    pub ssm_d: Option<Tensor>,      // D parameter (skip connection)
    pub ssm_delta: Option<Tensor>,  // Delta parameter
    pub ssm_delta_bias: Option<Tensor>,
    pub ssm_b: Option<Tensor>,     // B parameter
    pub ssm_c: Option<Tensor>,     // C parameter
    pub ssm_alpha: Option<Tensor>, // Alpha parameter

    // Hybrid: some layers have both attention and SSM
    pub post_attention_norm: Option<Tensor>,
    pub post_ffw_norm: Option<Tensor>,

    // Standard FFN (for hybrid models)
    pub ffn_norm: Option<Tensor>,
    pub ffn_gate: Option<Tensor>,
    pub ffn_down: Option<Tensor>,
    pub ffn_up: Option<Tensor>,
}

impl MambaLayer {
    pub fn from_weights(store: &WeightStore, layer_idx: usize) -> Self {
        let qwen_prefix = format!("blk.{}", layer_idx);

        MambaLayer {
            layer_idx,

            // Attention
            attention_norm: store
                .get(&format!("{}.attn_norm.weight", qwen_prefix))
                .cloned(),
            wq: store.get(&format!("{}.attn_q.weight", qwen_prefix)).cloned(),
            wk: store.get(&format!("{}.attn_k.weight", qwen_prefix)).cloned(),
            wv: store.get(&format!("{}.attn_v.weight", qwen_prefix)).cloned(),
            wo: store
                .get(&format!("{}.attn_output.weight", qwen_prefix))
                .cloned(),

            // SSM
            ssm_norm: store
                .get(&format!("{}.ssm_norm.weight", qwen_prefix))
                .cloned(),
            ssm_in: store
                .get(&format!("{}.ssm_in.weight", qwen_prefix))
                .cloned(),
            ssm_out: store
                .get(&format!("{}.ssm_out.weight", qwen_prefix))
                .cloned(),
            ssm_conv1d: store
                .get(&format!("{}.ssm_conv1d.weight", qwen_prefix))
                .cloned(),
            ssm_a: store.get(&format!("{}.ssm_a.weight", qwen_prefix)).cloned(),
            ssm_d: store.get(&format!("{}.ssm_d.weight", qwen_prefix)).cloned(),
            ssm_delta: store
                .get(&format!("{}.ssm_delta.weight", qwen_prefix))
                .cloned(),
            ssm_delta_bias: store
                .get(&format!("{}.ssm_delta.bias", qwen_prefix))
                .cloned(),
            ssm_b: store.get(&format!("{}.ssm_b.weight", qwen_prefix)).cloned(),
            ssm_c: store.get(&format!("{}.ssm_c.weight", qwen_prefix)).cloned(),
            ssm_alpha: store
                .get(&format!("{}.ssm_alpha.weight", qwen_prefix))
                .cloned(),

            // Hybrid norms
            post_attention_norm: store
                .get(&format!("{}.post_attention_norm.weight", qwen_prefix))
                .cloned(),
            post_ffw_norm: store
                .get(&format!("{}.post_ffw_norm.weight", qwen_prefix))
                .cloned(),

            // Standard FFN
            ffn_norm: store
                .get(&format!("{}.ffn_norm.weight", qwen_prefix))
                .cloned(),
            ffn_gate: store
                .get(&format!("{}.ffn_gate.weight", qwen_prefix))
                .cloned(),
            ffn_down: store
                .get(&format!("{}.ffn_down.weight", qwen_prefix))
                .cloned(),
            ffn_up: store
                .get(&format!("{}.ffn_up.weight", qwen_prefix))
                .cloned(),
        }
    }

    pub fn param_count(&self) -> usize {
        let mut count = 0usize;

        let tensors: Vec<Option<&Tensor>> = vec![
            self.attention_norm.as_ref(),
            self.wq.as_ref(),
            self.wk.as_ref(),
            self.wv.as_ref(),
            self.wo.as_ref(),
            self.ssm_norm.as_ref(),
            self.ssm_in.as_ref(),
            self.ssm_out.as_ref(),
            self.ssm_conv1d.as_ref(),
            self.ssm_a.as_ref(),
            self.ssm_d.as_ref(),
            self.ssm_delta.as_ref(),
            self.ssm_delta_bias.as_ref(),
            self.ssm_b.as_ref(),
            self.ssm_c.as_ref(),
            self.ssm_alpha.as_ref(),
            self.post_attention_norm.as_ref(),
            self.post_ffw_norm.as_ref(),
            self.ffn_norm.as_ref(),
            self.ffn_gate.as_ref(),
            self.ffn_down.as_ref(),
            self.ffn_up.as_ref(),
        ];

        for tensor in tensors.iter().flatten() {
            count += tensor.elem_count();
        }

        count
    }

    pub fn is_ssm(&self) -> bool {
        self.ssm_in.is_some()
            || self.ssm_out.is_some()
            || self.ssm_conv1d.is_some()
    }

    pub fn is_attention(&self) -> bool {
        self.wq.is_some() && self.wo.is_some()
    }

    pub fn is_hybrid(&self) -> bool {
        self.is_ssm() && self.is_attention()
    }

    pub fn is_complete(&self) -> bool {
        // SSM layer: need ssm_in, ssm_out
        // Attention layer: need wq, wo
        // Either is valid
        (self.is_ssm() && self.ssm_in.is_some() && self.ssm_out.is_some())
            || (self.is_attention() && self.wq.is_some() && self.wo.is_some())
    }
}
