// Mamba/SSM (State Space Model) layer definition
// Used by models like Qwen3.8-27B, Jamba, etc.

use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;

/// Mamba/SSM layer with state space model
pub struct MambaLayer {
    pub layer_idx: usize,

    // Attention (combined QKV for hybrid models)
    pub attention_norm: Option<Tensor>,
    pub attn_qkv: Option<Tensor>, // Combined QKV (SSM layers)
    pub attn_gate: Option<Tensor>, // Attention gate (SSM layers)
    pub wq: Option<Tensor>,       // Separate Q (pure attention)
    pub wk: Option<Tensor>,       // Separate K (pure attention)
    pub wv: Option<Tensor>,       // Separate V (pure attention)
    pub wo: Option<Tensor>,
    pub attn_q_norm: Option<Tensor>, // Q norm (pure attention)
    pub attn_k_norm: Option<Tensor>, // K norm (pure attention)

    // SSM (State Space Model)
    pub ssm_norm: Option<Tensor>,
    pub ssm_in: Option<Tensor>, // Input projection (some models)
    pub ssm_out: Option<Tensor>, // Output projection (always present)
    pub ssm_conv1d: Option<Tensor>, // 1D convolution
    pub ssm_a: Option<Tensor>,  // A parameter (no extension)
    pub ssm_dt: Option<Tensor>, // Delta parameter
    pub ssm_dt_bias: Option<Tensor>, // Delta bias
    pub ssm_b: Option<Tensor>,  // B parameter
    pub ssm_beta: Option<Tensor>, // Beta parameter (Qwen naming)
    pub ssm_c: Option<Tensor>,  // C parameter
    pub ssm_alpha: Option<Tensor>, // Alpha parameter (Qwen naming)
    // Legacy naming
    pub ssm_d: Option<Tensor>, // D parameter (skip connection)
    pub ssm_delta: Option<Tensor>, // Delta parameter (alt name)
    pub ssm_delta_bias: Option<Tensor>,

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
            attn_qkv: store
                .get(&format!("{}.attn_qkv.weight", qwen_prefix))
                .cloned(),
            attn_gate: store
                .get(&format!("{}.attn_gate.weight", qwen_prefix))
                .cloned(),
            wq: store.get(&format!("{}.attn_q.weight", qwen_prefix)).cloned(),
            wk: store.get(&format!("{}.attn_k.weight", qwen_prefix)).cloned(),
            wv: store.get(&format!("{}.attn_v.weight", qwen_prefix)).cloned(),
            wo: store
                .get(&format!("{}.attn_output.weight", qwen_prefix))
                .or_else(|| store.get(&format!("{}.wo.weight", qwen_prefix)))
                .cloned(),
            attn_q_norm: store
                .get(&format!("{}.attn_q_norm.weight", qwen_prefix))
                .cloned(),
            attn_k_norm: store
                .get(&format!("{}.attn_k_norm.weight", qwen_prefix))
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
            ssm_a: store
                .get(&format!("{}.ssm_a", qwen_prefix))
                .or_else(|| store.get(&format!("{}.ssm_a.weight", qwen_prefix)))
                .cloned(),
            ssm_dt: store
                .get(&format!("{}.ssm_dt.weight", qwen_prefix))
                .cloned(),
            ssm_dt_bias: store
                .get(&format!("{}.ssm_dt.bias", qwen_prefix))
                .cloned(),
            ssm_b: store.get(&format!("{}.ssm_b.weight", qwen_prefix)).cloned(),
            ssm_beta: store
                .get(&format!("{}.ssm_beta.weight", qwen_prefix))
                .cloned(),
            ssm_c: store.get(&format!("{}.ssm_c.weight", qwen_prefix)).cloned(),
            ssm_alpha: store
                .get(&format!("{}.ssm_alpha.weight", qwen_prefix))
                .cloned(),
            ssm_d: store.get(&format!("{}.ssm_d.weight", qwen_prefix)).cloned(),
            ssm_delta: store
                .get(&format!("{}.ssm_delta.weight", qwen_prefix))
                .cloned(),
            ssm_delta_bias: store
                .get(&format!("{}.ssm_delta.bias", qwen_prefix))
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
        let tensors: Vec<Option<&Tensor>> = vec![
            self.attention_norm.as_ref(),
            self.attn_qkv.as_ref(),
            self.attn_gate.as_ref(),
            self.wq.as_ref(),
            self.wk.as_ref(),
            self.wv.as_ref(),
            self.wo.as_ref(),
            self.attn_q_norm.as_ref(),
            self.attn_k_norm.as_ref(),
            self.ssm_norm.as_ref(),
            self.ssm_in.as_ref(),
            self.ssm_out.as_ref(),
            self.ssm_conv1d.as_ref(),
            self.ssm_a.as_ref(),
            self.ssm_dt.as_ref(),
            self.ssm_dt_bias.as_ref(),
            self.ssm_b.as_ref(),
            self.ssm_beta.as_ref(),
            self.ssm_c.as_ref(),
            self.ssm_alpha.as_ref(),
            self.ssm_d.as_ref(),
            self.ssm_delta.as_ref(),
            self.ssm_delta_bias.as_ref(),
            self.post_attention_norm.as_ref(),
            self.post_ffw_norm.as_ref(),
            self.ffn_norm.as_ref(),
            self.ffn_gate.as_ref(),
            self.ffn_down.as_ref(),
            self.ffn_up.as_ref(),
        ];

        tensors.iter().flatten().map(|t| t.elem_count()).sum()
    }

    pub fn is_ssm(&self) -> bool {
        // Qwen3.8-27B: ssm_out is present but ssm_in is NOT
        self.ssm_out.is_some()
            || self.ssm_conv1d.is_some()
            || self.ssm_a.is_some()
    }

    pub fn is_attention(&self) -> bool {
        // Combined QKV or separate Q+W
        (self.attn_qkv.is_some() || self.wq.is_some()) && self.wo.is_some()
    }

    pub fn is_hybrid(&self) -> bool {
        self.is_ssm() && self.is_attention()
    }

    pub fn is_complete(&self) -> bool {
        // SSM layer: need ssm_out (not ssm_in for Qwen models)
        // Attention layer: need (attn_qkv or wq), wo
        // Either is valid
        (self.is_ssm() && self.ssm_out.is_some())
            || (self.is_attention()
                && (self.attn_qkv.is_some() || self.wq.is_some())
                && self.wo.is_some())
    }
}
