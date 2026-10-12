// MoE (Mixture of Experts) layer definition
// Used by models like DeepSeek-Coder-V2, Mixtral, etc.

use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;

/// MoE layer with expert routing
pub struct MoeLayer {
    pub layer_idx: usize,

    // Attention (same as standard Llama)
    pub attention_norm: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub wo: Option<Tensor>,

    // FFN with MoE
    pub ffn_norm: Option<Tensor>,
    pub ffn_gate_inp: Option<Tensor>, // Router/gate for expert selection
    pub ffn_gate_shexp: Option<Tensor>, // Shared expert gate
    pub ffn_down_exps: Option<Tensor>, // Expert down projections
    pub ffn_gate_up_exps: Option<Tensor>, // Expert gate+up projections
    pub ffn_down_shexp: Option<Tensor>, // Shared expert down

    // Additional DeepSeek-Coder-V2 MoE components
    pub ffn_gate_exps: Option<Tensor>, // Expert gates
    pub ffn_up_shexp: Option<Tensor>,  // Shared expert up

    // Alternative naming
    pub ffn_experts: Option<Tensor>, // Some models store experts differently
}

impl MoeLayer {
    pub fn from_weights(store: &WeightStore, layer_idx: usize) -> Self {
        let llama_prefix = format!("layers.{}", layer_idx);
        let qwen_prefix = format!("blk.{}", layer_idx);

        MoeLayer {
            layer_idx,

            // Attention
            attention_norm: store
                .get(&format!("{}.attention_norm.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_norm.weight", qwen_prefix))
                })
                .cloned(),
            wq: store
                .get(&format!("{}.attention.wq.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_q.weight", qwen_prefix))
                })
                .cloned(),
            wk: store
                .get(&format!("{}.attention.wk.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_k.weight", qwen_prefix))
                })
                .cloned(),
            wv: store
                .get(&format!("{}.attention.wv.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_v.weight", qwen_prefix))
                })
                .cloned(),
            wo: store
                .get(&format!("{}.attention.wo.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_output.weight", qwen_prefix))
                })
                .cloned(),

            // FFN with MoE
            ffn_norm: store
                .get(&format!("{}.ffn_norm.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_norm.weight", qwen_prefix))
                })
                .cloned(),
            ffn_gate_inp: store
                .get(&format!("{}.feed_forward.gate_inp.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_gate_inp.weight", qwen_prefix))
                })
                .cloned(),
            ffn_gate_shexp: store
                .get(&format!(
                    "{}.feed_forward.gate_shexp.weight",
                    llama_prefix
                ))
                .or_else(|| {
                    store.get(&format!("{}.ffn_gate_shexp.weight", qwen_prefix))
                })
                .cloned(),
            ffn_down_exps: store
                .get(&format!("{}.feed_forward.down_exps.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_down_exps.weight", qwen_prefix))
                })
                .cloned(),
            ffn_gate_up_exps: store
                .get(&format!(
                    "{}.feed_forward.gate_up_exps.weight",
                    llama_prefix
                ))
                .or_else(|| {
                    store.get(&format!(
                        "{}.ffn_gate_up_exps.weight",
                        qwen_prefix
                    ))
                })
                .cloned(),
            ffn_down_shexp: store
                .get(&format!(
                    "{}.feed_forward.down_shexp.weight",
                    llama_prefix
                ))
                .or_else(|| {
                    store.get(&format!("{}.ffn_down_shexp.weight", qwen_prefix))
                })
                .cloned(),
            ffn_experts: store
                .get(&format!("{}.feed_forward.experts.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_experts.weight", qwen_prefix))
                })
                .cloned(),

            // Additional DeepSeek-Coder-V2 MoE components
            ffn_gate_exps: store
                .get(&format!("{}.ffn_gate_exps.weight", qwen_prefix))
                .cloned(),
            ffn_up_shexp: store
                .get(&format!("{}.ffn_up_shexp.weight", qwen_prefix))
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
            self.ffn_norm.as_ref(),
            self.ffn_gate_inp.as_ref(),
            self.ffn_gate_shexp.as_ref(),
            self.ffn_down_exps.as_ref(),
            self.ffn_gate_up_exps.as_ref(),
            self.ffn_down_shexp.as_ref(),
            self.ffn_experts.as_ref(),
        ];

        for tensor in tensors.iter().flatten() {
            count += tensor.elem_count();
        }

        count
    }

    pub fn is_moe(&self) -> bool {
        self.ffn_gate_inp.is_some()
            || self.ffn_down_exps.is_some()
            || self.ffn_gate_up_exps.is_some()
    }

    pub fn is_complete(&self) -> bool {
        self.attention_norm.is_some()
            && self.wq.is_some()
            && self.wo.is_some()
            && self.ffn_norm.is_some()
            && (self.is_moe()
                || (self.ffn_down_exps.is_some()
                    && self.ffn_gate_up_exps.is_some()))
    }
}
