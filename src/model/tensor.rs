// Generic tensor abstraction (format-agnostic)

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
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}
