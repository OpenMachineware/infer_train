// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

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

impl DataType {
    /// Size in bytes per element
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::Q4_0 => 18, // 16 bytes + 1 half scale
            DataType::Q4_1 => 20, // 16 bytes + 2 halves (scale + min)
            DataType::Q5_0 => 20,
            DataType::Q5_1 => 22,
            DataType::Q8_0 => 34, // 32 bytes + 1 half scale
            DataType::Q2_K => 84,
            DataType::Q3_K => 110,
            DataType::Q4_K => 144,
            DataType::Q5_K => 176,
            DataType::Q6_K => 210,
            DataType::IQ4_NL => 18,
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
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}
