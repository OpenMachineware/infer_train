// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Weight store: tensor manager for model weights

use crate::model::gguf::{GgufError, GgufFile, tensor_from_gguf};
use crate::model::tensor::{DataType, Tensor};
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
    Gguf(#[from] GgufError),

    #[error("Split file error: {0}")]
    Split(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
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

        // Check if this is a split file
        if gguf.is_split() {
            let (no, count) = gguf.split_info().unwrap();
            if no != 0 {
                return Err(WeightError::Split(format!(
                    "Must load first split part (part 0 of {}), got part {}",
                    count, no
                )));
            }
            return Self::load_all_splits(path, count);
        }

        let mut store =
            WeightStore { tensors: HashMap::new(), gguf: Some(gguf) };
        store.load_all_tensors()?;
        Ok(store)
    }

    fn load_all_splits(
        first_path: &str,
        count: usize,
    ) -> Result<Self, WeightError> {
        // Parse the split filename pattern: basename-NNNNN-of-MMMMM.gguf
        let (base_path, _suffix) = Self::parse_split_filename(first_path)?;

        let mut store = WeightStore { tensors: HashMap::new(), gguf: None };

        // Filenames are 1-indexed, metadata is 0-indexed
        for part in 1..=count {
            let part_path =
                format!("{}-{:05}-of-{:05}.gguf", base_path, part, count);
            let gguf = Arc::new(GgufFile::open(&part_path)?);

            for tensor_info in &gguf.tensors {
                if let Some(data) = gguf.get_tensor_data(&tensor_info.name) {
                    let tensor = tensor_from_gguf(tensor_info, data);
                    store.tensors.insert(tensor_info.name.clone(), tensor);
                }
            }

            if part == 1 {
                store.gguf = Some(gguf);
            }
        }

        Ok(store)
    }

    fn parse_split_filename(
        path: &str,
    ) -> Result<(String, String), WeightError> {
        // Pattern: basename-NNNNN-of-MMMMM.gguf
        let path = std::path::Path::new(path);
        let filename =
            path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
                WeightError::Split("Invalid filename".to_string())
            })?;

        // Must end with .gguf
        if !filename.ends_with(".gguf") {
            return Err(WeightError::Split(format!(
                "Not a GGUF file: {}",
                filename
            )));
        }

        // Remove .gguf extension
        let without_ext = &filename[..filename.len() - 5];

        // Pattern ends with -NNNNN-of-MMMMM
        let parts: Vec<&str> = without_ext.rsplitn(4, '-').collect();
        if parts.len() == 4 {
            // parts: [MMMMM, "of", NNNNN, basename...]
            if let Ok(_total) = parts[0].parse::<usize>() {
                if parts[1] == "of" {
                    if let Ok(_part_no) = parts[2].parse::<usize>() {
                        let base = parts[3];
                        let dir = path
                            .parent()
                            .map(|p| p.to_string_lossy())
                            .unwrap_or_default();
                        let base_path = if dir.is_empty() {
                            base.to_string()
                        } else {
                            format!("{}/{}", dir, base)
                        };
                        return Ok((base_path, filename.to_string()));
                    }
                }
            }
        }

        Err(WeightError::Split(format!(
            "Invalid split filename format: {}",
            filename
        )))
    }

    fn load_all_tensors(&mut self) -> Result<(), WeightError> {
        let gguf = self.gguf.as_ref().ok_or_else(|| WeightError::NotFound {
            name: "GGUF file".to_string(),
        })?;

        for tensor_info in &gguf.tensors {
            if let Some(data) = gguf.get_tensor_data(&tensor_info.name) {
                let tensor = tensor_from_gguf(tensor_info, data);
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

    /// Get iterator over tensor names
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
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
