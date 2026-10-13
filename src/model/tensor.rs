// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Generic tensor abstraction (format-agnostic)

use std::sync::Arc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: TensorData,
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub gpu_buffer: std::cell::RefCell<Option<std::rc::Rc<Buffer>>>,
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
    /// Block size in bytes for quantized formats, element size for FP formats
    pub fn block_size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::Q4_0 => 18,
            DataType::Q4_1 => 20,
            DataType::Q5_0 => 22,
            DataType::Q5_1 => 24,
            DataType::Q8_0 => 34,
            DataType::Q2_K => 84,
            DataType::Q3_K => 110,
            DataType::Q4_K => 144,
            DataType::Q5_K => 176,
            DataType::Q6_K => 210,
            DataType::IQ4_NL => 18,
        }
    }

    /// Elements per block (32 for standard Q, 256 for K-quant, 1 for FP)
    pub fn elems_per_block(&self) -> usize {
        match self {
            DataType::F32 => 1,
            DataType::F16 => 1,
            DataType::BF16 => 1,
            DataType::Q4_0 => 32,
            DataType::Q4_1 => 32,
            DataType::Q5_0 => 32,
            DataType::Q5_1 => 32,
            DataType::Q8_0 => 32,
            DataType::Q2_K => 256,
            DataType::Q3_K => 256,
            DataType::Q4_K => 256,
            DataType::Q5_K => 256,
            DataType::Q6_K => 256,
            DataType::IQ4_NL => 32,
        }
    }

    /// Check if this is a quantized format
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            DataType::Q4_0
                | DataType::Q4_1
                | DataType::Q5_0
                | DataType::Q5_1
                | DataType::Q8_0
                | DataType::Q2_K
                | DataType::Q3_K
                | DataType::Q4_K
                | DataType::Q5_K
                | DataType::Q6_K
                | DataType::IQ4_NL
        )
    }

    /// Calculate bytes needed for n elements
    pub fn bytes_for_elems(&self, n: usize) -> usize {
        let block_bytes = self.block_size();
        let elems_per_block = self.elems_per_block();
        let n_blocks = n.div_ceil(elems_per_block);
        n_blocks * block_bytes
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
