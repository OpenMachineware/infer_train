// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GGUF file format parser

use super::dtype::{GgufDtype, GgufError};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

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

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: GgufDtype,
    pub offset: u64,
}

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

impl GgufFile {
    pub fn open(path: &str) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let len = metadata.len() as usize;

        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = 0usize;

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

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mmap, &mut cursor)?;
            let value = read_metadata_value(&mmap, &mut cursor)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
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
            GgufDtype::Q4_0 => (elem_count / 32) * 18,
            GgufDtype::Q4_1 => (elem_count / 32) * 20,
            GgufDtype::Q5_0 => (elem_count / 32) * 22,
            GgufDtype::Q5_1 => (elem_count / 32) * 24,
            GgufDtype::Q8_0 => (elem_count / 32) * 34,
            GgufDtype::Q2_K => (elem_count / 256) * 84,
            GgufDtype::Q3_K => (elem_count / 256) * 110,
            GgufDtype::Q4_K => (elem_count / 256) * 144,
            GgufDtype::Q5_K => (elem_count / 256) * 176,
            GgufDtype::Q6_K => (elem_count / 256) * 210,
            _ => elem_count * 4,
        };
        bytes_per_elem
    }

    /// Check if this is a split GGUF file
    pub fn is_split(&self) -> bool {
        self.metadata.contains_key("split.count")
    }

    /// Get split info: (current_no, total_count)
    pub fn split_info(&self) -> Option<(usize, usize)> {
        let count = self.get_metadata_u64("split.count")?;
        let no = self.get_metadata_u64("split.no")?;
        Some((no as usize, count as usize))
    }

    /// Get metadata value as u64
    fn get_metadata_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key)? {
            MetadataValue::Uint64(v) => Some(*v),
            MetadataValue::Int64(v) => Some(*v as u64),
            MetadataValue::Uint32(v) => Some(*v as u64),
            MetadataValue::Int32(v) => Some(*v as u64),
            MetadataValue::Uint16(v) => Some(*v as u64),
            MetadataValue::Int16(v) => Some(*v as u64),
            MetadataValue::Uint8(v) => Some(*v as u64),
            MetadataValue::Int8(v) => Some(*v as u64),
            _ => None,
        }
    }
}

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
            let val = data[*cursor];
            *cursor += 1;
            Ok(MetadataValue::Uint8(val))
        }
        1 => {
            let val = data[*cursor] as i8;
            *cursor += 1;
            Ok(MetadataValue::Int8(val))
        }
        2 => {
            let val = u16::from_le_bytes(
                data[*cursor..*cursor + 2].try_into().unwrap(),
            );
            *cursor += 2;
            Ok(MetadataValue::Uint16(val))
        }
        3 => {
            let val = i16::from_le_bytes(
                data[*cursor..*cursor + 2].try_into().unwrap(),
            );
            *cursor += 2;
            Ok(MetadataValue::Int16(val))
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
        9 => {
            let type_id = read_u32(data, cursor)?;
            let count = read_u64(data, cursor)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_metadata_value_by_type(data, cursor, type_id)?);
            }
            Ok(MetadataValue::Array(arr))
        }
        10 => {
            let val = read_u64(data, cursor)?;
            Ok(MetadataValue::Uint64(val))
        }
        11 => {
            let val = i64::from_le_bytes(
                data[*cursor..*cursor + 8].try_into().unwrap(),
            );
            *cursor += 8;
            Ok(MetadataValue::Int64(val))
        }
        12 => {
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
