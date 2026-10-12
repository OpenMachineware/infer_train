// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Hybrid MoE + SSM layer definition
// Used by models like Qwen3.6-35B (MoE + SSM hybrid)

use crate::model::tensor::Tensor;
use crate::model::weights::WeightStore;

/// Hybrid MoE + SSM layer
pub struct HybridMoESsmLayer {
    pub layer_idx: usize,

    // Attention (combined QKV)
    pub attention_norm: Option<Tensor>,
    pub attn_qkv: Option<Tensor>,
    pub attn_gate: Option<Tensor>,
    pub wo: Option<Tensor>,

    // SSM (State Space Model)
    pub ssm_norm: Option<Tensor>,
    pub ssm_in: Option<Tensor>,
    pub ssm_out: Option<Tensor>,
    pub ssm_conv1d: Option<Tensor>,
    pub ssm_a: Option<Tensor>,
    pub ssm_dt: Option<Tensor>,
    pub ssm_dt_bias: Option<Tensor>,
    pub ssm_b: Option<Tensor>,
    pub ssm_c: Option<Tensor>,

    // MoE (Mixture of Experts)
    pub ffn_norm: Option<Tensor>,
    pub ffn_gate_inp: Option<Tensor>,
    pub ffn_gate_shexp: Option<Tensor>,
    pub ffn_down_exps: Option<Tensor>,
    pub ffn_gate_up_exps: Option<Tensor>,
    pub ffn_down_shexp: Option<Tensor>,

    // Post norms
    pub post_attention_norm: Option<Tensor>,
    pub post_ffw_norm: Option<Tensor>,
}

impl HybridMoESsmLayer {
    pub fn from_weights(store: &WeightStore, layer_idx: usize) -> Self {
        let prefix = format!("blk.{}", layer_idx);

        HybridMoESsmLayer {
            layer_idx,

            // Attention
            attention_norm: store
                .get(&format!("{}.attn_norm.weight", prefix))
                .cloned(),
            attn_qkv: store
                .get(&format!("{}.attn_qkv.weight", prefix))
                .cloned(),
            attn_gate: store
                .get(&format!("{}.attn_gate.weight", prefix))
                .cloned(),
            wo: store.get(&format!("{}.wo.weight", prefix)).cloned(),

            // SSM
            ssm_norm: store
                .get(&format!("{}.ssm_norm.weight", prefix))
                .cloned(),
            ssm_in: store.get(&format!("{}.ssm_in.weight", prefix)).cloned(),
            ssm_out: store.get(&format!("{}.ssm_out.weight", prefix)).cloned(),
            ssm_conv1d: store
                .get(&format!("{}.ssm_conv1d.weight", prefix))
                .cloned(),
            ssm_a: store
                .get(&format!("{}.ssm_a", prefix))
                .or_else(|| store.get(&format!("{}.ssm_a.weight", prefix)))
                .cloned(),
            ssm_dt: store.get(&format!("{}.ssm_dt.weight", prefix)).cloned(),
            ssm_dt_bias: store.get(&format!("{}.ssm_dt.bias", prefix)).cloned(),
            ssm_b: store.get(&format!("{}.ssm_b.weight", prefix)).cloned(),
            ssm_c: store.get(&format!("{}.ssm_c.weight", prefix)).cloned(),

            // MoE
            ffn_norm: store
                .get(&format!("{}.ffn_norm.weight", prefix))
                .cloned(),
            ffn_gate_inp: store
                .get(&format!("{}.ffn_gate_inp.weight", prefix))
                .cloned(),
            ffn_gate_shexp: store
                .get(&format!("{}.ffn_gate_shexp.weight", prefix))
                .cloned(),
            ffn_down_exps: store
                .get(&format!("{}.ffn_down_exps.weight", prefix))
                .cloned(),
            ffn_gate_up_exps: store
                .get(&format!("{}.ffn_gate_up_exps.weight", prefix))
                .cloned(),
            ffn_down_shexp: store
                .get(&format!("{}.ffn_down_shexp.weight", prefix))
                .cloned(),

            // Post norms
            post_attention_norm: store
                .get(&format!("{}.post_attention_norm.weight", prefix))
                .cloned(),
            post_ffw_norm: store
                .get(&format!("{}.post_ffw_norm.weight", prefix))
                .cloned(),
        }
    }

    pub fn param_count(&self) -> usize {
        let tensors: Vec<Option<&Tensor>> = vec![
            self.attention_norm.as_ref(),
            self.attn_qkv.as_ref(),
            self.attn_gate.as_ref(),
            self.wo.as_ref(),
            self.ssm_norm.as_ref(),
            self.ssm_in.as_ref(),
            self.ssm_out.as_ref(),
            self.ssm_conv1d.as_ref(),
            self.ssm_a.as_ref(),
            self.ssm_dt.as_ref(),
            self.ssm_dt_bias.as_ref(),
            self.ssm_b.as_ref(),
            self.ssm_c.as_ref(),
            self.ffn_norm.as_ref(),
            self.ffn_gate_inp.as_ref(),
            self.ffn_gate_shexp.as_ref(),
            self.ffn_down_exps.as_ref(),
            self.ffn_gate_up_exps.as_ref(),
            self.ffn_down_shexp.as_ref(),
            self.post_attention_norm.as_ref(),
            self.post_ffw_norm.as_ref(),
        ];

        tensors.iter().flatten().map(|t| t.elem_count()).sum()
    }

    pub fn is_ssm(&self) -> bool {
        self.ssm_in.is_some() || self.ssm_out.is_some()
    }

    pub fn is_moe(&self) -> bool {
        self.ffn_down_exps.is_some() || self.ffn_gate_inp.is_some()
    }

    pub fn is_attention(&self) -> bool {
        self.attn_qkv.is_some() || self.wo.is_some()
    }

    pub fn is_complete(&self) -> bool {
        // Need at least one functional component
        let has_attention = self.attn_qkv.is_some();
        let has_ssm = self.ssm_in.is_some() && self.ssm_out.is_some();
        let has_moe =
            self.ffn_down_exps.is_some() || self.ffn_gate_inp.is_some();

        has_attention || has_ssm || has_moe
    }
}
