// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Unified model loader with architecture detection

use crate::model::config::ModelConfig;
use crate::model::gguf::GgufFile;
use crate::model::hybrid::HybridMoESsmLayer;
use crate::model::llama::{LlamaLayer, LlamaModel};
use crate::model::mamba::MambaLayer;
use crate::model::moe::MoeLayer;
use crate::model::tensor::Tensor;
use crate::model::weights::WeightStore;

/// Model architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    MoE,
    Mamba,
    Hybrid,
    Unknown,
}

/// Unified model with multiple architecture support
pub struct Model {
    pub config: ModelConfig,
    pub architecture: Architecture,

    // Common components
    pub tok_embeddings: Option<Tensor>,
    pub norm: Option<Tensor>,
    pub output: Option<Tensor>,

    // Architecture-specific layers
    pub llama_layers: Vec<LlamaLayer>,
    pub moe_layers: Vec<MoeLayer>,
    pub mamba_layers: Vec<MambaLayer>,
    pub hybrid_layers: Vec<HybridMoESsmLayer>,
}

impl Model {
    pub fn from_gguf(path: &str) -> Result<Self, String> {
        let gguf = GgufFile::open(path).map_err(|e| e.to_string())?;
        let config = ModelConfig::from_gguf(&gguf);
        let store = WeightStore::from_gguf(path).map_err(|e| e.to_string())?;

        Ok(Self::from_weights(&store, config))
    }

    pub fn from_weights(store: &WeightStore, config: ModelConfig) -> Self {
        let architecture = Self::detect_architecture(store, &config);

        // Load embeddings
        let tok_embeddings = store
            .get("tok_embeddings.weight")
            .or_else(|| store.get("token_embd.weight"))
            .cloned();

        // Load output norm
        let norm = store
            .get("norm.weight")
            .or_else(|| store.get("output_norm.weight"))
            .cloned();

        // Load output projection
        let output = store
            .get("output.weight")
            .or_else(|| store.get("lm_head.weight"))
            .cloned();

        // Load layers based on architecture
        let (llama_layers, moe_layers, mamba_layers, hybrid_layers) =
            match architecture {
                Architecture::Llama => {
                    let layers: Vec<LlamaLayer> = (0..config.n_layers)
                        .map(|i| LlamaLayer::from_weights(store, i))
                        .collect();
                    (layers, vec![], vec![], vec![])
                }
                Architecture::MoE => {
                    let layers: Vec<MoeLayer> = (0..config.n_layers)
                        .map(|i| MoeLayer::from_weights(store, i))
                        .collect();
                    (vec![], layers, vec![], vec![])
                }
                Architecture::Mamba => {
                    let layers: Vec<MambaLayer> = (0..config.n_layers)
                        .map(|i| MambaLayer::from_weights(store, i))
                        .collect();
                    (vec![], vec![], layers, vec![])
                }
                Architecture::Hybrid => {
                    let layers: Vec<HybridMoESsmLayer> = (0..config.n_layers)
                        .map(|i| HybridMoESsmLayer::from_weights(store, i))
                        .collect();
                    (vec![], vec![], vec![], layers)
                }
                Architecture::Unknown => {
                    // Fallback: try Llama
                    let layers: Vec<LlamaLayer> = (0..config.n_layers)
                        .map(|i| LlamaLayer::from_weights(store, i))
                        .collect();
                    (layers, vec![], vec![], vec![])
                }
            };

        Model {
            config,
            architecture,
            tok_embeddings,
            norm,
            output,
            llama_layers,
            moe_layers,
            mamba_layers,
            hybrid_layers,
        }
    }

    fn detect_architecture(
        store: &WeightStore,
        config: &ModelConfig,
    ) -> Architecture {
        // Check first few layers for architecture hints
        for i in 0..std::cmp::min(5, config.n_layers) {
            let prefix = format!("blk.{}.", i);

            // Check for MoE + SSM hybrid (has both ffn_down_exps and ssm_in)
            let has_moe =
                store.get(&format!("{}ffn_down_exps.weight", prefix)).is_some()
                    || store
                        .get(&format!("{}ffn_gate_inp.weight", prefix))
                        .is_some();
            let has_ssm = store
                .get(&format!("{}ssm_in.weight", prefix))
                .is_some()
                || store.get(&format!("{}ssm_out.weight", prefix)).is_some();

            if has_moe && has_ssm {
                return Architecture::Hybrid;
            }

            // Check for pure MoE
            if has_moe {
                return Architecture::MoE;
            }

            // Check for pure Mamba/SSM
            if has_ssm {
                return Architecture::Mamba;
            }
        }

        // Default: Llama-style
        Architecture::Llama
    }

    pub fn layer_count(&self) -> usize {
        self.llama_layers.len()
            + self.moe_layers.len()
            + self.mamba_layers.len()
            + self.hybrid_layers.len()
    }

    pub fn complete_layers(&self) -> usize {
        let llama =
            self.llama_layers.iter().filter(|l| l.is_complete()).count();
        let moe = self.moe_layers.iter().filter(|l| l.is_complete()).count();
        let mamba =
            self.mamba_layers.iter().filter(|l| l.is_complete()).count();
        let hybrid =
            self.hybrid_layers.iter().filter(|l| l.is_complete()).count();
        llama + moe + mamba + hybrid
    }

    pub fn param_count(&self) -> usize {
        let mut count = 0usize;

        if let Some(ref t) = self.tok_embeddings {
            count += t.elem_count();
        }
        if let Some(ref t) = self.norm {
            count += t.elem_count();
        }
        if let Some(ref t) = self.output {
            count += t.elem_count();
        }

        for layer in &self.llama_layers {
            count += layer.param_count();
        }
        for layer in &self.moe_layers {
            count += layer.param_count();
        }
        for layer in &self.mamba_layers {
            count += layer.param_count();
        }
        for layer in &self.hybrid_layers {
            count += layer.param_count();
        }

        count
    }
}
