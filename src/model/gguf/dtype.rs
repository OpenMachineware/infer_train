// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GGUF data types enum

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4, // Unused
    Q4_3 = 5, // Unused
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
}

impl TryFrom<u32> for GgufDtype {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgufDtype::F32),
            1 => Ok(GgufDtype::F16),
            2 => Ok(GgufDtype::Q4_0),
            3 => Ok(GgufDtype::Q4_1),
            6 => Ok(GgufDtype::Q5_0),
            7 => Ok(GgufDtype::Q5_1),
            8 => Ok(GgufDtype::Q8_0),
            9 => Ok(GgufDtype::Q8_1),
            10 => Ok(GgufDtype::Q2_K),
            11 => Ok(GgufDtype::Q3_K),
            12 => Ok(GgufDtype::Q4_K),
            13 => Ok(GgufDtype::Q5_K),
            14 => Ok(GgufDtype::Q6_K),
            15 => Ok(GgufDtype::Q8_K),
            16 => Ok(GgufDtype::IQ2_XXS),
            17 => Ok(GgufDtype::IQ2_XS),
            18 => Ok(GgufDtype::IQ3_XXS),
            19 => Ok(GgufDtype::IQ1_S),
            20 => Ok(GgufDtype::IQ4_NL),
            21 => Ok(GgufDtype::IQ3_S),
            22 => Ok(GgufDtype::IQ2_S),
            23 => Ok(GgufDtype::IQ4_XS),
            24 => Ok(GgufDtype::I8),
            25 => Ok(GgufDtype::I16),
            26 => Ok(GgufDtype::I32),
            27 => Ok(GgufDtype::I64),
            28 => Ok(GgufDtype::F64),
            29 => Ok(GgufDtype::BF16),
            _ => Err(GgufError::UnknownDtype(value)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("Invalid GGUF magic: expected {expected:#x}, got {actual:#x}")]
    InvalidMagic { expected: u32, actual: u32 },

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Unknown data type: {0}")]
    UnknownDtype(u32),

    #[error("Unknown metadata type: {0}")]
    UnknownMetadataType(u32),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("Invalid tensor name at index {index}")]
    InvalidTensorName { index: usize },

    #[error("Tensor '{name}' not found")]
    TensorNotFound { name: String },
}
