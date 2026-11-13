// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG builder from GGUF model

use crate::dag::ComputeGraph;
use crate::model::GgufFile;
use crate::ops::{CoreOperator, QuantType};

/// Build compute graph from GGUF model
pub struct DagBuilder {
    graph: ComputeGraph,
}

impl DagBuilder {
    pub fn new() -> Self {
        Self { graph: ComputeGraph::new() }
    }

    /// Build graph for a transformer layer
    pub fn build_layer(
        &mut self,
        layer_idx: usize,
        hidden_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        max_seq_len: usize,
    ) {
        let prefix = format!("layers.{}", layer_idx);
        let hs = hidden_size as u64;
        let q_dim = (n_heads * head_dim) as u64;
        let kv_dim = (n_kv_heads * head_dim) as u64;
        let is = intermediate_size as u64;

        // Input
        let input = self.graph.add_input(
            &format!("{}.input", prefix),
            vec![1, hs],
            QuantType::F32,
        );

        // RMS Norm for attention
        let attn_norm = self.graph.add_op(
            &format!("{}.attn_norm", prefix),
            CoreOperator::RmsNorm { hidden_size, eps: 1e-5 },
            vec![input],
            vec![1, hs],
            QuantType::F32,
        );

        // Q projection
        let q_proj = self.graph.add_weight(
            &format!("{}.q_proj.weight", prefix),
            vec![q_dim, hs],
            QuantType::Q4_0,
            vec![],
        );

        // K projection
        let k_proj = self.graph.add_weight(
            &format!("{}.k_proj.weight", prefix),
            vec![kv_dim, hs],
            QuantType::Q4_0,
            vec![],
        );

        // V projection
        let v_proj = self.graph.add_weight(
            &format!("{}.v_proj.weight", prefix),
            vec![kv_dim, hs],
            QuantType::Q4_0,
            vec![],
        );

        // Attention
        let _attn = self.graph.add_op(
            &format!("{}.attn", prefix),
            CoreOperator::FlashAttention {
                n_heads,
                n_kv_heads,
                head_dim,
                max_seq_len,
            },
            vec![attn_norm, q_proj, k_proj, v_proj],
            vec![1, hs],
            QuantType::F32,
        );

        // O projection
        let _o_proj = self.graph.add_weight(
            &format!("{}.o_proj.weight", prefix),
            vec![hs, q_dim],
            QuantType::Q4_0,
            vec![],
        );

        // FFN norm
        let _ffn_norm = self.graph.add_op(
            &format!("{}.ffn_norm", prefix),
            CoreOperator::RmsNorm { hidden_size, eps: 1e-5 },
            vec![input],
            vec![1, hs],
            QuantType::F32,
        );

        // FFN (SwiGLU)
        let _ffn = self.graph.add_op(
            &format!("{}.ffn", prefix),
            CoreOperator::SwiGLU { hidden_size, intermediate_size },
            vec![input],
            vec![1, hs],
            QuantType::F32,
        );

        // Gate and up projections for FFN
        let _gate_proj = self.graph.add_weight(
            &format!("{}.gate_proj.weight", prefix),
            vec![is, hs],
            QuantType::Q4_0,
            vec![],
        );

        let _up_proj = self.graph.add_weight(
            &format!("{}.up_proj.weight", prefix),
            vec![is, hs],
            QuantType::Q4_0,
            vec![],
        );

        let _down_proj = self.graph.add_weight(
            &format!("{}.down_proj.weight", prefix),
            vec![hs, is],
            QuantType::Q4_0,
            vec![],
        );
    }

    /// Build complete graph from GGUF
    pub fn build_from_gguf(gguf: &GgufFile) -> Result<ComputeGraph, String> {
        let mut builder = Self::new();

        // Extract model config from metadata
        let n_layers =
            gguf.get_meta::<i64>("llama.block_count").unwrap_or(32) as usize;
        let hidden_size = gguf
            .get_meta::<i64>("llama.embedding_length")
            .unwrap_or(4096) as usize;
        let n_heads = gguf
            .get_meta::<i64>("llama.attention.head_count")
            .unwrap_or(32) as usize;
        let n_kv_heads = gguf
            .get_meta::<i64>("llama.attention.head_count_kv")
            .unwrap_or(n_heads as i64) as usize;
        let head_dim = hidden_size / n_heads;
        let intermediate_size = gguf
            .get_meta::<i64>("llama.feed_forward_length")
            .unwrap_or(11008) as usize;
        let max_seq_len = gguf
            .get_meta::<i64>("llama.context_length")
            .unwrap_or(4096) as usize;
        let vocab_size =
            gguf.get_meta::<i64>("llama.vocab_size").unwrap_or(32000) as u64;
        let hs = hidden_size as u64;

        // Add embedding weights
        let _embed = builder.graph.add_weight(
            "token_embd.weight",
            vec![vocab_size, hs],
            QuantType::Q4_0,
            vec![],
        );

        // Build layers
        for i in 0..n_layers {
            builder.build_layer(
                i,
                hidden_size,
                n_heads,
                n_kv_heads,
                head_dim,
                intermediate_size,
                max_seq_len,
            );
        }

        // Final norm
        let _final_norm = builder.graph.add_op(
            "output_norm",
            CoreOperator::RmsNorm { hidden_size, eps: 1e-5 },
            vec![],
            vec![1, hs],
            QuantType::F32,
        );

        // Output projection
        let _output = builder.graph.add_weight(
            "output.weight",
            vec![vocab_size, hs],
            QuantType::Q4_0,
            vec![],
        );

        builder.graph.build_execution_order()?;
        Ok(builder.graph)
    }

    pub fn graph(self) -> ComputeGraph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_layer() {
        let mut builder = DagBuilder::new();
        builder.build_layer(0, 4096, 32, 32, 128, 11008, 4096);

        let graph = builder.graph;
        assert!(graph.tensors.len() > 0);
    }
}
