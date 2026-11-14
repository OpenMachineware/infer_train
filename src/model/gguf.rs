// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GGUF file format parser
//
// GGUF binary format structure:
// 1. Magic "GGUF" (4 bytes)
// 2. Version (uint32)
// 3. Number of tensors (int64)
// 4. Number of KV pairs (int64)
// 5. KV pairs
// 6. Tensor info
// 7. Tensor data (aligned)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::types::QuantType;

// GGUF magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

// Default alignment for tensor data
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// GGUF value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

// GGUF value
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GgufArrayType, Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

// Array element type (for GgufValue::Array)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufArrayType(pub GgufType);

// Tensor metadata
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub quant_type: QuantType,
    pub offset: u64,
}

// Parsed GGUF file
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub kv: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
    pub data: Vec<u8>,
}

impl GgufFile {
    /// Parse a GGUF file from path
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        let mut reader = BufReader::new(file);
        Self::parse(&mut reader)
    }

    /// Parse GGUF from reader
    pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Self, String> {
        // 1. Magic
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "Invalid GGUF magic: expected 0x{:08x}, got 0x{:08x}",
                GGUF_MAGIC, magic
            ));
        }

        // 2. Version
        let version = read_u32(reader)?;

        // 3. Number of tensors
        let n_tensors = read_i64(reader)? as usize;

        // 4. Number of KV pairs
        let n_kv = read_i64(reader)? as usize;

        // 5. Parse KV pairs
        let mut kv = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = read_string(reader)?;
            let value = read_value(reader)?;
            kv.insert(key, value);
        }

        // Get alignment from metadata
        let alignment = match kv.get("general.alignment") {
            Some(GgufValue::Uint32(v)) => *v as u64,
            Some(GgufValue::Int32(v)) => *v as u64,
            Some(GgufValue::Uint64(v)) => *v,
            Some(GgufValue::Int64(v)) => *v as u64,
            _ => GGUF_DEFAULT_ALIGNMENT,
        };

        // 6. Parse tensor info
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(reader)?;
            let n_dims = read_u32(reader)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_i64(reader)? as u64);
            }
            let quant_type = read_quant_type(reader)?;
            let offset = read_u64(reader)?;

            tensors.push(GgufTensorInfo { name, dims, quant_type, offset });
        }

        // Calculate data offset (aligned)
        let current_pos = reader
            .stream_position()
            .map_err(|e| format!("Stream position error: {}", e))?;
        let data_offset = align_to(current_pos, alignment);

        // Read tensor data
        reader
            .seek(SeekFrom::Start(data_offset))
            .map_err(|e| format!("Seek error: {}", e))?;
        let data_len = reader
            .seek(SeekFrom::End(0))
            .map_err(|e| format!("Seek error: {}", e))?
            as usize
            - data_offset as usize;
        reader
            .seek(SeekFrom::Start(data_offset))
            .map_err(|e| format!("Seek error: {}", e))?;

        let mut data = vec![0u8; data_len];
        reader
            .read_exact(&mut data)
            .map_err(|e| format!("Read tensor data error: {}", e))?;

        Ok(GgufFile { version, alignment, kv, tensors, data_offset, data })
    }

    /// Get a metadata value by key
    pub fn get_meta<T: FromGgufValue>(&self, key: &str) -> Option<T> {
        self.kv.get(key).and_then(|v| T::from_gguf(v))
    }

    /// Get tensor data slice by name
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        for tensor in &self.tensors {
            if tensor.name == name {
                let start = tensor.offset as usize;
                let size = Self::tensor_size(tensor);
                return Some(&self.data[start..start + size]);
            }
        }
        None
    }

    /// Calculate tensor size in bytes
    pub fn tensor_size(tensor: &GgufTensorInfo) -> usize {
        let n_elements: u64 = tensor.dims.iter().product();
        let bytes_per_element = match tensor.quant_type {
            QuantType::F32 => 4,
            QuantType::F16 => 2,
            QuantType::Q4_0 => 1, // block-wise
            QuantType::Q4_1 => 1,
            QuantType::Q5_0 => 1,
            QuantType::Q5_1 => 1,
            QuantType::Q8_0 => 1,
            QuantType::Q8_1 => 1,
            QuantType::Q2_K => 1,
            QuantType::Q3_K => 1,
            QuantType::Q4_K => 1,
            QuantType::Q5_K => 1,
            QuantType::Q6_K => 1,
            QuantType::Q8_K => 1,
            QuantType::BF16 => 2,
        };
        // For quantized types, block-based calculation would be needed
        // This is a simplified version
        (n_elements * bytes_per_element) as usize
    }
}

// Helper functions

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64, String> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    Ok(i64::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, String> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Read string error: {}", e))?;
    String::from_utf8(buf).map_err(|e| format!("Invalid UTF-8 string: {}", e))
}

fn read_value<R: Read>(reader: &mut R) -> Result<GgufValue, String> {
    let type_id = read_i32(reader)?;
    let gguf_type = GgufType::try_from(type_id)?;
    match gguf_type {
        GgufType::Uint8 => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint8(buf[0]))
        }
        GgufType::Int8 => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int8(buf[0] as i8))
        }
        GgufType::Uint16 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint16(u16::from_le_bytes(buf)))
        }
        GgufType::Int16 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
        }
        GgufType::Uint32 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint32(u32::from_le_bytes(buf)))
        }
        GgufType::Int32 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int32(i32::from_le_bytes(buf)))
        }
        GgufType::Float32 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
        }
        GgufType::Bool => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Bool(buf[0] != 0))
        }
        GgufType::String => {
            let s = read_string(reader)?;
            Ok(GgufValue::String(s))
        }
        GgufType::Array => {
            let elem_type_id = read_i32(reader)?;
            let elem_type = GgufType::try_from(elem_type_id)?;
            let n = read_u64(reader)? as usize;
            let mut elements = Vec::with_capacity(n);
            for _ in 0..n {
                elements.push(read_value_of_type(reader, elem_type)?);
            }
            Ok(GgufValue::Array(GgufArrayType(elem_type), elements))
        }
        GgufType::Uint64 => {
            let mut buf = [0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint64(u64::from_le_bytes(buf)))
        }
        GgufType::Int64 => {
            let mut buf = [0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
        }
        GgufType::Float64 => {
            let mut buf = [0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
        }
    }
}

fn read_value_of_type<R: Read>(
    reader: &mut R,
    gguf_type: GgufType,
) -> Result<GgufValue, String> {
    match gguf_type {
        GgufType::Uint8 => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint8(buf[0]))
        }
        GgufType::Int8 => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int8(buf[0] as i8))
        }
        GgufType::Uint16 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint16(u16::from_le_bytes(buf)))
        }
        GgufType::Int16 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
        }
        GgufType::Uint32 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint32(u32::from_le_bytes(buf)))
        }
        GgufType::Int32 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int32(i32::from_le_bytes(buf)))
        }
        GgufType::Float32 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
        }
        GgufType::Bool => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Bool(buf[0] != 0))
        }
        GgufType::String => {
            let s = read_string(reader)?;
            Ok(GgufValue::String(s))
        }
        GgufType::Uint64 => {
            let mut buf = [0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Uint64(u64::from_le_bytes(buf)))
        }
        GgufType::Int64 => {
            let mut buf = [0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
        }
        GgufType::Float64 => {
            let mut buf = [0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
        }
        _ => Err(format!("Unsupported array element type: {:?}", gguf_type)),
    }
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(|e| format!("Read error: {}", e))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_quant_type<R: Read>(reader: &mut R) -> Result<QuantType, String> {
    let type_id = read_i32(reader)?;
    QuantType::try_from(type_id)
}

fn align_to(offset: u64, alignment: u64) -> u64 {
    ((offset + alignment - 1) / alignment) * alignment
}

impl TryFrom<i32> for GgufType {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgufType::Uint8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::Uint16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::Uint32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::Uint64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            _ => Err(format!("Unknown GGUF type: {}", value)),
        }
    }
}

// Trait for extracting typed values from GgufValue
pub trait FromGgufValue: Sized {
    fn from_gguf(value: &GgufValue) -> Option<Self>;
}

impl FromGgufValue for i32 {
    fn from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::Int32(v) => Some(*v),
            GgufValue::Int64(v) => Some(*v as i32),
            _ => None,
        }
    }
}

impl FromGgufValue for i64 {
    fn from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::Int32(v) => Some(*v as i64),
            GgufValue::Int64(v) => Some(*v),
            _ => None,
        }
    }
}

impl FromGgufValue for u32 {
    fn from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }
}

impl FromGgufValue for u64 {
    fn from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
}

impl FromGgufValue for f32 {
    fn from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

impl FromGgufValue for String {
    fn from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::String(v) => Some(v.clone()),
            _ => None,
        }
    }
}

impl FromGgufValue for bool {
    fn from_gguf(value: &GgufValue) -> Option<Self> {
        match value {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_type_conversion() {
        assert_eq!(GgufType::try_from(0), Ok(GgufType::Uint8));
        assert_eq!(GgufType::try_from(5), Ok(GgufType::Int32));
        assert_eq!(GgufType::try_from(8), Ok(GgufType::String));
        assert_eq!(
            GgufType::try_from(-1),
            Err("Unknown GGUF type: -1".to_string())
        );
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 32), 0);
        assert_eq!(align_to(1, 32), 32);
        assert_eq!(align_to(31, 32), 32);
        assert_eq!(align_to(32, 32), 32);
        assert_eq!(align_to(33, 32), 64);
    }
}
