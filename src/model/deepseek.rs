// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// DeepSeek-V2/V3 style MoE layer with Multi-Query Attention
// Uses MQA (attn_kv_a_mqa, attn_kv_b) instead of separate wk/wv

use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;

/// DeepSeek-V2/V3 MoE layer with Multi-Query Attention
pub struct DeepSeekMoeLayer {
    pub layer_idx: usize,

    // Multi-Query Attention
    pub attn_norm: Option<Tensor>,
    pub attn_q: Option<Tensor>,
    pub attn_kv_a_mqa: Option<Tensor>, // KV A projection (Multi-Query)
    pub attn_kv_a_norm: Option<Tensor>, // KV A normalization
    pub attn_kv_b: Option<Tensor>,      // KV B projection
    pub attn_output: Option<Tensor>,

    // FFN - can be standard or MoE
    pub ffn_norm: Option<Tensor>,
    pub ffn_gate: Option<Tensor>,  // Standard FFN gate
    pub ffn_up: Option<Tensor>,    // Standard FFN up
    pub ffn_down: Option<Tensor>,  // Standard FFN down

    // MoE components (present in MoE layers)
    pub ffn_gate_inp: Option<Tensor>,      // Expert router
    pub ffn_gate_inp_shexp: Option<Tensor>, // Shared expert router
    pub ffn_gate_exps: Option<Tensor>,      // Expert gates
    pub ffn_gate_up_exps: Option<Tensor>,   // Expert gate+up
    pub ffn_down_exps: Option<Tensor>,      // Expert down
    pub ffn_gate_shexp: Option<Tensor>,     // Shared expert gate
    pub ffn_up_shexp: Option<Tensor>,       // Shared expert up
    pub ffn_down_shexp: Option<Tensor>,     // Shared expert down
}

impl DeepSeekMoeLayer {
    pub fn from_weights(store: &WeightStore, layer_idx: usize) -> Self {
        let prefix = format!("blk.{}.", layer_idx);

        DeepSeekMoeLayer {
            layer_idx,

            // Multi-Query Attention
            attn_norm: store.get(&format!("{}attn_norm.weight", prefix)).cloned(),
            attn_q: store.get(&format!("{}attn_q.weight", prefix)).cloned(),
            attn_kv_a_mqa: store.get(&format!("{}attn_kv_a_mqa.weight", prefix)).cloned(),
            attn_kv_a_norm: store.get(&format!("{}attn_kv_a_norm.weight", prefix)).cloned(),
            attn_kv_b: store.get(&format!("{}attn_kv_b.weight", prefix)).cloned(),
            attn_output: store.get(&format!("{}attn_output.weight", prefix)).cloned(),

            // Standard FFN
            ffn_norm: store.get(&format!("{}ffn_norm.weight", prefix)).cloned(),
            ffn_gate: store.get(&format!("{}ffn_gate.weight", prefix)).cloned(),
            ffn_up: store.get(&format!("{}ffn_up.weight", prefix)).cloned(),
            ffn_down: store.get(&format!("{}ffn_down.weight", prefix)).cloned(),

            // MoE components
            ffn_gate_inp: store.get(&format!("{}ffn_gate_inp.weight", prefix)).cloned(),
            ffn_gate_inp_shexp: store.get(&format!("{}ffn_gate_inp_shexp.weight", prefix)).cloned(),
            ffn_gate_exps: store.get(&format!("{}ffn_gate_exps.weight", prefix)).cloned(),
            ffn_gate_up_exps: store.get(&format!("{}ffn_gate_up_exps.weight", prefix)).cloned(),
            ffn_down_exps: store.get(&format!("{}ffn_down_exps.weight", prefix)).cloned(),
            ffn_gate_shexp: store.get(&format!("{}ffn_gate_shexp.weight", prefix)).cloned(),
            ffn_up_shexp: store.get(&format!("{}ffn_up_shexp.weight", prefix)).cloned(),
            ffn_down_shexp: store.get(&format!("{}ffn_down_shexp.weight", prefix)).cloned(),
        }
    }

    pub fn param_count(&self) -> usize {
        let mut count = 0usize;

        let tensors: Vec<Option<&Tensor>> = vec![
            self.attn_norm.as_ref(),
            self.attn_q.as_ref(),
            self.attn_kv_a_mqa.as_ref(),
            self.attn_kv_a_norm.as_ref(),
            self.attn_kv_b.as_ref(),
            self.attn_output.as_ref(),
            self.ffn_norm.as_ref(),
            self.ffn_gate.as_ref(),
            self.ffn_up.as_ref(),
            self.ffn_down.as_ref(),
            self.ffn_gate_inp.as_ref(),
            self.ffn_gate_inp_shexp.as_ref(),
            self.ffn_gate_exps.as_ref(),
            self.ffn_gate_up_exps.as_ref(),
            self.ffn_down_exps.as_ref(),
            self.ffn_gate_shexp.as_ref(),
            self.ffn_up_shexp.as_ref(),
            self.ffn_down_shexp.as_ref(),
        ];

        for tensor in tensors.iter().flatten() {
            count += tensor.elem_count();
        }

        count
    }

    pub fn is_moe(&self) -> bool {
        self.ffn_gate_inp.is_some() || self.ffn_down_exps.is_some()
    }

    pub fn is_mqa(&self) -> bool {
        self.attn_kv_a_mqa.is_some()
    }

    pub fn is_complete(&self) -> bool {
        // Attention: need attn_q, attn_kv_a_mqa or attn_output
        let has_attention = self.attn_norm.is_some()
            && self.attn_q.is_some()
            && self.attn_output.is_some();

        // FFN: either standard or MoE
        let has_ffn = self.ffn_norm.is_some()
            && (self.ffn_down.is_some()
                || self.ffn_down_exps.is_some());

        has_attention && has_ffn
    }
}
