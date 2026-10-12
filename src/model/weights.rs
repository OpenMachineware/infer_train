// Weight store: tensor manager for model weights

use crate::model::gguf::GgufFile;
use crate::model::tensor::{DataType, Tensor, TensorData};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("Tensor '{name}' not found")]
    NotFound { name: String },

    #[error("Invalid tensor shape: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Invalid data type: expected {expected:?}, got {actual:?}")]
    DtypeMismatch { expected: DataType, actual: DataType },

    #[error("GGUF error: {0}")]
    Gguf(#[from] crate::model::gguf::GgufError),
}

pub struct WeightStore {
    tensors: HashMap<String, Tensor>,
    gguf: Option<Arc<GgufFile>>,
}

impl WeightStore {
    pub fn new() -> Self {
        WeightStore { tensors: HashMap::new(), gguf: None }
    }

    pub fn from_gguf(path: &str) -> Result<Self, WeightError> {
        let gguf = Arc::new(GgufFile::open(path)?);
        let mut store =
            WeightStore { tensors: HashMap::new(), gguf: Some(gguf) };
        store.load_all_tensors()?;
        Ok(store)
    }

    fn load_all_tensors(&mut self) -> Result<(), WeightError> {
        let gguf = self.gguf.as_ref().ok_or_else(|| WeightError::NotFound {
            name: "GGUF file".to_string(),
        })?;

        for tensor_info in &gguf.tensors {
            if let Some(data) = gguf.get_tensor_data(&tensor_info.name) {
                let tensor = Tensor::from_gguf(tensor_info, data);
                self.tensors.insert(tensor_info.name.clone(), tensor);
            }
        }

        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn get_required(&self, name: &str) -> Result<&Tensor, WeightError> {
        self.tensors
            .get(name)
            .ok_or_else(|| WeightError::NotFound { name: name.to_string() })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.bytes().len()).sum()
    }
}

impl Default for WeightStore {
    fn default() -> Self {
        Self::new()
    }
}
