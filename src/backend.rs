// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Backend trait definition

pub mod arm;
pub mod cuda;
pub mod metal;

use crate::types::{CoreOperator, OpError, QuantType};

// Backend ID type
pub type BackendId = usize;

// Tensor reference - supports both FP32 and quantized data
#[derive(Debug, Clone)]
pub struct TensorRef {
    /// Data pointer (can be FP32, FP16, or quantized blocks)
    pub data: *const u8,
    /// Data type
    pub dtype: TensorType,
    /// Dimensions (up to 4D)
    pub dims: [usize; 4],
    /// Number of dimensions
    pub ndim: usize,
    /// Strides in bytes
    pub strides: [usize; 4],
}

// Mutable tensor reference
#[derive(Debug)]
pub struct TensorRefMut {
    pub data: *mut u8,
    pub dtype: TensorType,
    pub dims: [usize; 4],
    pub ndim: usize,
    pub strides: [usize; 4],
}

// Tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum TensorType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

impl TensorRef {
    pub fn from_f32(data: &[f32]) -> Self {
        let len = data.len();
        Self {
            data: data.as_ptr() as *const u8,
            dtype: TensorType::F32,
            dims: [len, 1, 1, 1],
            ndim: 1,
            strides: [4, 0, 0, 0],
        }
    }

    pub fn from_f32_2d(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            data: data.as_ptr() as *const u8,
            dtype: TensorType::F32,
            dims: [cols, rows, 1, 1],
            ndim: 2,
            strides: [4, cols * 4, 0, 0],
        }
    }

    pub fn len(&self) -> usize {
        let mut len = 1;
        for i in 0..self.ndim {
            len *= self.dims[i];
        }
        len
    }

    pub fn nbytes(&self) -> usize {
        self.len() * self.dtype.size()
    }
}

impl TensorRefMut {
    pub fn from_f32_mut(data: &mut [f32]) -> Self {
        let len = data.len();
        Self {
            data: data.as_mut_ptr() as *mut u8,
            dtype: TensorType::F32,
            dims: [len, 1, 1, 1],
            ndim: 1,
            strides: [4, 0, 0, 0],
        }
    }

    pub fn from_f32_mut_2d(data: &mut [f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            data: data.as_mut_ptr() as *mut u8,
            dtype: TensorType::F32,
            dims: [cols, rows, 1, 1],
            ndim: 2,
            strides: [4, cols * 4, 0, 0],
        }
    }

    pub fn len(&self) -> usize {
        let mut len = 1;
        for i in 0..self.ndim {
            len *= self.dims[i];
        }
        len
    }
}

impl TensorType {
    pub fn size(&self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::F16 => 2,
            TensorType::BF16 => 2,
            TensorType::Q4_0 => 18, // BlockQ4_0
            TensorType::Q4_1 => 20,
            TensorType::Q5_0 => 22,
            TensorType::Q5_1 => 24,
            TensorType::Q8_0 => 34,
            TensorType::Q8_1 => 36,
            TensorType::Q2_K => 84,
            TensorType::Q3_K => 110,
            TensorType::Q4_K => 144,
            TensorType::Q5_K => 176,
            TensorType::Q6_K => 210,
            TensorType::Q8_K => 292,
        }
    }
}

impl From<QuantType> for TensorType {
    fn from(qt: QuantType) -> Self {
        match qt {
            QuantType::F32 => TensorType::F32,
            QuantType::F16 => TensorType::F16,
            QuantType::BF16 => TensorType::BF16,
            QuantType::Q4_0 => TensorType::Q4_0,
            QuantType::Q4_1 => TensorType::Q4_1,
            QuantType::Q5_0 => TensorType::Q5_0,
            QuantType::Q5_1 => TensorType::Q5_1,
            QuantType::Q8_0 => TensorType::Q8_0,
            QuantType::Q8_1 => TensorType::Q8_1,
            QuantType::Q2_K => TensorType::Q2_K,
            QuantType::Q3_K => TensorType::Q3_K,
            QuantType::Q4_K => TensorType::Q4_K,
            QuantType::Q5_K => TensorType::Q5_K,
            QuantType::Q6_K => TensorType::Q6_K,
            QuantType::Q8_K => TensorType::Q8_K,
            // IQ types fallback to Q4_0 for now
            _ => TensorType::Q4_0,
        }
    }
}

// Backend trait
pub trait Backend: Send + Sync {
    /// Backend identifier
    fn id(&self) -> usize;
    fn name(&self) -> &str;

    /// Memory info
    fn available_memory(&self) -> usize;

    /// Capability query
    fn supports_op(&self, op: &CoreOperator) -> bool;
    fn supports_quant(&self, quant: QuantType) -> bool;

    /// Execute operator
    fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRefMut],
    ) -> Result<(), OpError>;
}

// Backend registry
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    pub fn find(&self, op: &CoreOperator) -> Option<&dyn Backend> {
        self.backends.iter().find(|b| b.supports_op(op)).map(|b| b.as_ref())
    }
}
