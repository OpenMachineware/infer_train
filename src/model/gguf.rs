// GGUF file format parser
// Based on: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use thiserror::Error;

// GGUF magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

// GGUF data types
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

// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

// Tensor info
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: GgufDtype,
    pub offset: u64,
}

// GGUF file structure
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    mmap: Mmap,
}

#[derive(Debug, Error)]
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

impl GgufFile {
    pub fn open(path: &str) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let len = metadata.len() as usize;

        // Safety: file is opened read-only and won't be modified
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = 0usize;

        // Read header
        let magic = read_u32(&mmap, &mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic {
                expected: GGUF_MAGIC,
                actual: magic,
            });
        }

        let version = read_u32(&mmap, &mut cursor)?;

        let tensor_count = read_u64(&mmap, &mut cursor)?;
        let metadata_kv_count = read_u64(&mmap, &mut cursor)?;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mmap, &mut cursor)?;
            let value = read_metadata_value(&mmap, &mut cursor)?;
            metadata.insert(key, value);
        }

        // Read tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = read_string(&mmap, &mut cursor)?;
            let n_dims = read_u32(&mmap, &mut cursor)?;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mmap, &mut cursor)?);
            }

            let dtype_val = read_u32(&mmap, &mut cursor)?;
            let dtype = GgufDtype::try_from(dtype_val)?;
            let offset = read_u64(&mmap, &mut cursor)?;

            tensors.push(TensorInfo { name, n_dims, dims, dtype, offset });
        }

        let data_offset = cursor as u64;

        Ok(GgufFile {
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensors,
            data_offset,
            mmap,
        })
    }

    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.iter().find(|t| t.name == name)?;

        let start = self.data_offset as usize + tensor.offset as usize;
        let size = self.tensor_size(tensor);

        if start + size <= self.mmap.len() {
            Some(&self.mmap[start..start + size])
        } else {
            None
        }
    }

    fn tensor_size(&self, tensor: &TensorInfo) -> usize {
        let elem_count: usize = tensor.dims.iter().product::<u64>() as usize;
        let bytes_per_elem = match tensor.dtype {
            GgufDtype::F32 => 4,
            GgufDtype::F16 => 2,
            GgufDtype::BF16 => 2,
            GgufDtype::Q4_0 => {
                // 32 values per block: 2 bytes scale + 16 bytes quants = 18 bytes
                (elem_count / 32) * 18
            }
            GgufDtype::Q4_1 => {
                // 32 values per block: 2 + 2 + 16 = 20 bytes
                (elem_count / 32) * 20
            }
            GgufDtype::Q5_0 => {
                // 32 values per block: 2 + 4 + 16 = 22 bytes
                (elem_count / 32) * 22
            }
            GgufDtype::Q5_1 => {
                // 32 values per block: 2 + 2 + 4 + 16 = 24 bytes
                (elem_count / 32) * 24
            }
            GgufDtype::Q8_0 => {
                // 32 values per block: 2 + 32 = 34 bytes
                (elem_count / 32) * 34
            }
            GgufDtype::Q2_K => (elem_count / 256) * 84,
            GgufDtype::Q3_K => (elem_count / 256) * 110,
            GgufDtype::Q4_K => (elem_count / 256) * 144,
            GgufDtype::Q5_K => (elem_count / 256) * 176,
            GgufDtype::Q6_K => (elem_count / 256) * 210,
            _ => elem_count * 4, // Default to F32 size
        };
        bytes_per_elem
    }
}

// Helper functions for reading GGUF binary format
fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32, GgufError> {
    if *cursor + 4 > data.len() {
        return Err(GgufError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Unexpected end of file",
        )));
    }
    let val =
        u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64, GgufError> {
    if *cursor + 8 > data.len() {
        return Err(GgufError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Unexpected end of file",
        )));
    }
    let val =
        u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(val)
}

fn read_string(data: &[u8], cursor: &mut usize) -> Result<String, GgufError> {
    let len = read_u64(data, cursor)? as usize;
    if *cursor + len > data.len() {
        return Err(GgufError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Unexpected end of file",
        )));
    }
    let s = std::str::from_utf8(&data[*cursor..*cursor + len])?;
    *cursor += len;
    Ok(s.to_string())
}

fn read_metadata_value(
    data: &[u8],
    cursor: &mut usize,
) -> Result<MetadataValue, GgufError> {
    let type_id = read_u32(data, cursor)?;

    match type_id {
        0 => {
            // UINT8
            let val = data[*cursor];
            *cursor += 1;
            Ok(MetadataValue::Uint8(val))
        }
        1 => {
            // INT8
            let val = data[*cursor] as i8;
            *cursor += 1;
            Ok(MetadataValue::Int8(val))
        }
        2 => {
            // UINT16
            let val = u16::from_le_bytes(
                data[*cursor..*cursor + 2].try_into().unwrap(),
            );
            *cursor += 2;
            Ok(MetadataValue::Uint16(val))
        }
        3 => {
            // INT16
            let val = i16::from_le_bytes(
                data[*cursor..*cursor + 2].try_into().unwrap(),
            );
            *cursor += 2;
            Ok(MetadataValue::Int16(val))
        }
        4 => {
            // UINT32
            let val = read_u32(data, cursor)?;
            Ok(MetadataValue::Uint32(val))
        }
        5 => {
            // INT32
            let val = i32::from_le_bytes(
                data[*cursor..*cursor + 4].try_into().unwrap(),
            );
            *cursor += 4;
            Ok(MetadataValue::Int32(val))
        }
        6 => {
            // FLOAT32
            let val = f32::from_le_bytes(
                data[*cursor..*cursor + 4].try_into().unwrap(),
            );
            *cursor += 4;
            Ok(MetadataValue::Float32(val))
        }
        7 => {
            // BOOL
            let val = data[*cursor] != 0;
            *cursor += 1;
            Ok(MetadataValue::Bool(val))
        }
        8 => {
            // STRING
            let s = read_string(data, cursor)?;
            Ok(MetadataValue::String(s))
        }
        9 => {
            // ARRAY
            let type_id = read_u32(data, cursor)?;
            let count = read_u64(data, cursor)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_metadata_value_by_type(data, cursor, type_id)?);
            }
            Ok(MetadataValue::Array(arr))
        }
        10 => {
            // UINT64
            let val = read_u64(data, cursor)?;
            Ok(MetadataValue::Uint64(val))
        }
        11 => {
            // INT64
            let val = i64::from_le_bytes(
                data[*cursor..*cursor + 8].try_into().unwrap(),
            );
            *cursor += 8;
            Ok(MetadataValue::Int64(val))
        }
        12 => {
            // FLOAT64
            let val = f64::from_le_bytes(
                data[*cursor..*cursor + 8].try_into().unwrap(),
            );
            *cursor += 8;
            Ok(MetadataValue::Float64(val))
        }
        _ => Err(GgufError::UnknownMetadataType(type_id)),
    }
}

fn read_metadata_value_by_type(
    data: &[u8],
    cursor: &mut usize,
    type_id: u32,
) -> Result<MetadataValue, GgufError> {
    match type_id {
        0 => {
            let val = data[*cursor];
            *cursor += 1;
            Ok(MetadataValue::Uint8(val))
        }
        1 => {
            let val = data[*cursor] as i8;
            *cursor += 1;
            Ok(MetadataValue::Int8(val))
        }
        4 => {
            let val = read_u32(data, cursor)?;
            Ok(MetadataValue::Uint32(val))
        }
        5 => {
            let val = i32::from_le_bytes(
                data[*cursor..*cursor + 4].try_into().unwrap(),
            );
            *cursor += 4;
            Ok(MetadataValue::Int32(val))
        }
        6 => {
            let val = f32::from_le_bytes(
                data[*cursor..*cursor + 4].try_into().unwrap(),
            );
            *cursor += 4;
            Ok(MetadataValue::Float32(val))
        }
        7 => {
            let val = data[*cursor] != 0;
            *cursor += 1;
            Ok(MetadataValue::Bool(val))
        }
        8 => {
            let s = read_string(data, cursor)?;
            Ok(MetadataValue::String(s))
        }
        _ => Err(GgufError::UnknownMetadataType(type_id)),
    }
}
