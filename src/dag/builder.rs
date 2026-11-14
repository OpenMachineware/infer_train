// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// DAG builder from GGUF model

use crate::dag::{ComputeGraph, Op};
use crate::model::GgufFile;
use crate::types::QuantType;

// Build compute graph from GGUF model
pub struct DagBuilder {
    graph: ComputeGraph,
}

impl DagBuilder {
    pub fn new() -> Self {
        Self { graph: ComputeGraph::new() }
    }

    // Build graph by parsing GGUF tensor names
    pub fn build_from_gguf(gguf: &GgufFile) -> Result<ComputeGraph, String> {
        let mut builder = Self::new();

        // Extract model config
        let config = ModelConfig::from_gguf(gguf);

        // Load all weight tensors from GGUF
        builder.load_weights(gguf, &config);

        // Build input (embedding lookup)
        builder.build_embedding(&config);

        // Build layers dynamically
        for i in 0..config.n_layers {
            builder.build_layer(i, &config);
        }

        // Build output
        builder.build_output(&config);

        builder.graph.build_execution_order();
        Ok(builder.graph)
    }

    fn load_weights(&mut self, gguf: &GgufFile, _config: &ModelConfig) {
        for tensor in &gguf.tensors {
            // Determine tensor role by name
            let dims: Vec<u64> =
                tensor.dims.iter().map(|&d| d as u64).collect();
            self.graph.add_weight(
                &tensor.name,
                &dims,
                tensor.quant_type,
                tensor.offset,
            );
        }
    }

    fn build_embedding(&mut self, config: &ModelConfig) {
        // Input tokens
        let tokens = self.graph.add_tensor("tokens", &[1], QuantType::F32);

        // Embedding lookup
        if self.graph.get_tensor_by_name("token_embd.weight").is_some() {
            let _embedded = self.graph.add_op(
                "embedded",
                Op::Embedding,
                &[tokens],
                &[1, config.hidden_size as u64],
                QuantType::F32,
            );
        }
    }

    fn build_layer(&mut self, layer_idx: usize, _config: &ModelConfig) {
        let _prefix = format!("blk.{}", layer_idx);
        // Build attention and FFN blocks
        // This is simplified - real implementation would:
        // 1. Find weight tensors by name
        // 2. Connect them through operations
        // 3. Handle residual connections
    }

    fn build_output(&mut self, config: &ModelConfig) {
        // Final norm and output projection
        if self.graph.get_tensor_by_name("output.weight").is_some() {
            let output = self.graph.add_op(
                "logits",
                Op::MulMat,
                &[],
                &[config.vocab_size as u64, config.hidden_size as u64],
                QuantType::F32,
            );
            self.graph.outputs.push(output);
        }
    }

    pub fn graph(self) -> ComputeGraph {
        self.graph
    }
}

// Model configuration extracted from GGUF
struct ModelConfig {
    n_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
}

impl ModelConfig {
    fn from_gguf(gguf: &GgufFile) -> Self {
        Self {
            n_layers: gguf.get_meta::<i64>("llama.block_count").unwrap_or(32)
                as usize,
            hidden_size: gguf
                .get_meta::<i64>("llama.embedding_length")
                .unwrap_or(4096) as usize,
            vocab_size: gguf
                .get_meta::<i64>("llama.vocab_size")
                .unwrap_or(32000) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let builder = DagBuilder::new();
        let graph = builder.graph();
        assert_eq!(graph.tensors.len(), 0);
    }
}
