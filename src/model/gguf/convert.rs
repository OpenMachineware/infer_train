// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Conversion from GGUF format to generic Tensor

use super::dtype::GgufDtype;
use super::parser::TensorInfo;
use crate::model::tensor::{DataType, Tensor, TensorData};
use std::sync::Arc;

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
            _ => DataType::F32,
        }
    }
}

pub fn tensor_from_gguf(info: &TensorInfo, data: &[u8]) -> Tensor {
    Tensor {
        name: info.name.clone(),
        shape: info.dims.iter().map(|&d| d as usize).collect(),
        dtype: DataType::from(info.dtype),
        data: TensorData::Mmap(Arc::from(data.to_vec().into_boxed_slice())),
    }
}
