// Tensor abstraction for model weights

use crate::model::gguf::{GgufDtype, GgufFile, TensorInfo};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: TensorData,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    IQ4_NL,
}

impl From<GgufDtype> for DataType {
    fn from(dtype: GgufDtype) -> Self {
        match dtype {
            GgufDtype::F32 => DataType::F32,
            GgufDtype::F16 => DataType::F16,
            GgufDtype::BF16 => DataType::BF16,
            GgufDtype::Q4_0 => DataType::Q4_0,
            GgufDtype::Q4_1 => DataType::Q4_1,
            GgufDtype::Q5_0 => DataType::Q5_0,
            GgufDtype::Q5_1 => DataType::Q5_1,
            GgufDtype::Q8_0 => DataType::Q8_0,
            GgufDtype::Q2_K => DataType::Q2_K,
            GgufDtype::Q3_K => DataType::Q3_K,
            GgufDtype::Q4_K => DataType::Q4_K,
            GgufDtype::Q5_K => DataType::Q5_K,
            GgufDtype::Q6_K => DataType::Q6_K,
            GgufDtype::IQ4_NL => DataType::IQ4_NL,
            _ => DataType::F32, // Default fallback
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorData {
    Owned(Vec<u8>),
    Mmap(Arc<[u8]>),
}

impl TensorData {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            TensorData::Owned(v) => v,
            TensorData::Mmap(m) => m,
        }
    }
}

impl Tensor {
    pub fn from_gguf(info: &TensorInfo, data: &[u8]) -> Self {
        Tensor {
            name: info.name.clone(),
            shape: info.dims.iter().map(|&d| d as usize).collect(),
            dtype: DataType::from(info.dtype),
            data: TensorData::Mmap(Arc::from(data.to_vec().into_boxed_slice())),
        }
    }

    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}
