// Llama model architecture definition

use crate::model::config::ModelConfig;
use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;
use std::collections::HashMap;

/// Llama model layers
pub struct LlamaModel {
    pub config: ModelConfig,

    // Embedding
    pub tok_embeddings: Option<Tensor>,

    // Transformer blocks
    pub layers: Vec<LlamaLayer>,

    // Output
    pub norm: Option<Tensor>,
    pub output: Option<Tensor>,
}

pub struct LlamaLayer {
    pub layer_idx: usize,

    // Attention
    pub attention_norm: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub wo: Option<Tensor>,

    // FFN
    pub ffn_norm: Option<Tensor>,
    pub w1: Option<Tensor>, // gate
    pub w2: Option<Tensor>, // down
    pub w3: Option<Tensor>, // up
}

impl LlamaModel {
    pub fn from_weights(store: &WeightStore, config: ModelConfig) -> Self {
        let mut model = LlamaModel {
            config,
            tok_embeddings: None,
            layers: Vec::new(),
            norm: None,
            output: None,
        };

        // Load embeddings
        model.tok_embeddings = store.get("tok_embeddings.weight").cloned();

        // Load layers
        for i in 0..model.config.n_layers {
            let layer = LlamaLayer::from_weights(store, i);
            model.layers.push(layer);
        }

        // Load output norm
        model.norm = store.get("norm.weight").cloned();

        // Load output projection
        model.output = store.get("output.weight").cloned();

        model
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn param_count(&self) -> usize {
        let mut count = 0usize;

        if let Some(ref t) = self.tok_embeddings {
            count += t.elem_count();
        }
        for layer in &self.layers {
            count += layer.param_count();
        }
        if let Some(ref t) = self.norm {
            count += t.elem_count();
        }
        if let Some(ref t) = self.output {
            count += t.elem_count();
        }

        count
    }
}

impl LlamaLayer {
    pub fn from_weights(store: &WeightStore, layer_idx: usize) -> Self {
        let prefix = format!("layers.{}", layer_idx);

        LlamaLayer {
            layer_idx,

            // Attention
            attention_norm: store
                .get(&format!("{}.attention_norm.weight", prefix))
                .cloned(),
            wq: store.get(&format!("{}.attention.wq.weight", prefix)).cloned(),
            wk: store.get(&format!("{}.attention.wk.weight", prefix)).cloned(),
            wv: store.get(&format!("{}.attention.wv.weight", prefix)).cloned(),
            wo: store.get(&format!("{}.attention.wo.weight", prefix)).cloned(),

            // FFN
            ffn_norm: store
                .get(&format!("{}.ffn_norm.weight", prefix))
                .cloned(),
            w1: store
                .get(&format!("{}.feed_forward.w1.weight", prefix))
                .cloned(),
            w2: store
                .get(&format!("{}.feed_forward.w2.weight", prefix))
                .cloned(),
            w3: store
                .get(&format!("{}.feed_forward.w3.weight", prefix))
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
            self.w1.as_ref(),
            self.w2.as_ref(),
            self.w3.as_ref(),
        ];

        for tensor in tensors.iter().flatten() {
            count += tensor.elem_count();
        }

        count
    }

    pub fn is_complete(&self) -> bool {
        self.attention_norm.is_some()
            && self.wq.is_some()
            && self.wk.is_some()
            && self.wv.is_some()
            && self.wo.is_some()
            && self.ffn_norm.is_some()
            && self.w1.is_some()
            && self.w2.is_some()
            && self.w3.is_some()
    }
}

/// Alternative tensor name patterns (different GGUF naming conventions)
pub fn find_tensor_name(
    store: &WeightStore,
    patterns: &[&str],
) -> Option<String> {
    for pattern in patterns {
        if store.get(pattern).is_some() {
            return Some(pattern.to_string());
        }
    }
    None
}

/// Build tensor name mapping from GGUF file
pub fn build_tensor_mapping(store: &WeightStore) -> HashMap<String, String> {
    let mut mapping = HashMap::new();

    for name in store.tensor_names() {
        // Map common variations
        if name.contains("tok_embeddings") || name.contains("embed_tokens") {
            mapping.insert("tok_embeddings".to_string(), name.clone());
        }
        if name.starts_with("layers.") && name.contains("attention_norm") {
            if let Some(layer_idx) = extract_layer_index(name) {
                mapping.insert(
                    format!("layers.{}.attention_norm", layer_idx),
                    name.clone(),
                );
            }
        }
    }

    mapping
}

fn extract_layer_index(name: &str) -> Option<usize> {
    if let Some(start) = name.find("layers.") {
        let rest = &name[start + 7..];
        if let Some(end) = rest.find('.') {
            if let Ok(idx) = rest[..end].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}
