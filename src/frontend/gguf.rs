// src/frontend/gguf.rs

use std::collections::HashMap;

use gguf_rs_lib::reader::file_reader::open_gguf_file;
use gguf_rs_lib::reader::GGUFFileReader;
use gguf_rs_lib::tensor::info::TensorInfo;
use gguf_rs_lib::format::types::GGUFTensorType;
use gguf_rs_lib::format::metadata::Metadata;
use gguf_rs_lib::format::metadata::MetadataValue;
use gguf_rs_lib::builder::GGUFBuilder;

use half::{f16, bf16};

use pyo3::prelude::*;

use crate::ir::dag::{DagGraph, Op, AttrValue, TensorType as DagTensorType, DataType};

// ============================================================
// 常量
// ============================================================

const SUPPORTED_ARCHITECTURES: &[&str] = &["llama", "mistral", "qwen", "phi"];

// ============================================================
// 量化类型
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_K,
    Q6_K,
}

impl QuantType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(QuantType::F32),
            "F16" => Some(QuantType::F16),
            "BF16" => Some(QuantType::BF16),
            "Q8_0" => Some(QuantType::Q8_0),
            "Q4_0" => Some(QuantType::Q4_0),
            "Q4_K" => Some(QuantType::Q4_K),
            "Q6_K" => Some(QuantType::Q6_K),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            QuantType::F32 => "F32",
            QuantType::F16 => "F16",
            QuantType::BF16 => "BF16",
            QuantType::Q8_0 => "Q8_0",
            QuantType::Q4_0 => "Q4_0",
            QuantType::Q4_K => "Q4_K",
            QuantType::Q6_K => "Q6_K",
        }
    }

    pub fn to_gguf_type(&self) -> GGUFTensorType {
        match self {
            QuantType::F32 => GGUFTensorType::F32,
            QuantType::F16 => GGUFTensorType::F16,
            QuantType::BF16 => GGUFTensorType::BF16,
            QuantType::Q8_0 => GGUFTensorType::Q8_0,
            QuantType::Q4_0 => GGUFTensorType::Q4_0,
            QuantType::Q4_K => GGUFTensorType::Q4_K,
            QuantType::Q6_K => GGUFTensorType::Q6_K,
        }
    }
}

// ============================================================
// 类型别名 (简化泛型)
// ============================================================

type GGUFFileReaderType = GGUFFileReader<std::io::BufReader<std::fs::File>>;

// ============================================================
// 元数据结构
// ============================================================

#[derive(Debug, Clone)]
pub struct GGUFModelMetadata {
    pub architecture: String,
    pub context_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

// ============================================================
// Tensor 数据结构
// ============================================================

#[derive(Debug, Clone)]
pub struct GGUFTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub quant_type: GGUFTensorType,
}

// ============================================================
// 读取元数据
// ============================================================

fn read_metadata(reader: &GGUFFileReaderType) -> Result<GGUFModelMetadata, String> {
    let metadata = reader.metadata();

    fn get_str(metadata: &Metadata, key: &str) -> Result<String, String> {
        metadata.get(key)
            .and_then(|v| match v {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .ok_or_else(|| format!("Missing metadata: {}", key))
    }

    fn get_int(metadata: &Metadata, key: &str) -> Result<usize, String> {
        metadata.get(key)
            .and_then(|v| match v {
                MetadataValue::U64(u) => Some(*u as usize),
                MetadataValue::I64(i) => Some(*i as usize),
                _ => None,
            })
            .ok_or_else(|| format!("Missing metadata: {}", key))
    }

    fn get_float(metadata: &Metadata, key: &str) -> Result<f32, String> {
        metadata.get(key)
            .and_then(|v| match v {
                MetadataValue::F32(f) => Some(*f),
                MetadataValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            .ok_or_else(|| format!("Missing metadata: {}", key))
    }

    Ok(GGUFModelMetadata {
        architecture: get_str(metadata, "general.architecture")?,
        context_length: get_int(metadata, "llama.context_length")
            .or_else(|_| get_int(metadata, "llama.max_seq_len"))
            .unwrap_or(4096),
        hidden_size: get_int(metadata, "llama.embedding_length")
            .or_else(|_| get_int(metadata, "llama.hidden_size"))
            .unwrap_or(4096),
        num_layers: get_int(metadata, "llama.block_count")
            .or_else(|_| get_int(metadata, "llama.num_layers"))
            .unwrap_or(32),
        num_heads: get_int(metadata, "llama.attention_head_count")
            .or_else(|_| get_int(metadata, "llama.num_heads"))
            .unwrap_or(32),
        num_key_value_heads: get_int(metadata, "llama.attention_head_count_kv")
            .or_else(|_| get_int(metadata, "llama.num_key_value_heads"))
            .unwrap_or(8),
        vocab_size: get_int(metadata, "llama.vocab_size")
            .or_else(|_| get_int(metadata, "llama.n_vocab"))
            .unwrap_or(32000),
        intermediate_size: get_int(metadata, "llama.feed_forward_length")
            .or_else(|_| get_int(metadata, "llama.intermediate_size"))
            .unwrap_or(11008),
        hidden_act: get_str(metadata, "llama.hidden_act")
            .or_else(|_| Ok::<String, String>("silu".to_string()))
            .unwrap_or_else(|_| "silu".to_string()),
        rms_norm_eps: get_float(metadata, "llama.attention_norm_eps")
            .or_else(|_| get_float(metadata, "llama.rms_norm_eps"))
            .unwrap_or(1e-5),
        rope_theta: get_float(metadata, "llama.rope_theta")
            .or_else(|_| Ok::<f32, String>(10000.0))
            .unwrap_or(10000.0),
    })
}

// ============================================================
// 提取 Tensor 数据 (反量化)
// ============================================================

fn extract_tensor_data(tensor_info: &TensorInfo) -> Result<Vec<u8>, String> {
    let raw_data = tensor_info.data()
        .ok_or("Tensor has no data")?;
    let raw_bytes = raw_data.as_slice();

    match tensor_info.tensor_type {
        GGUFTensorType::F32 => Ok(raw_bytes.to_vec()),
        GGUFTensorType::F16 => {
            let mut f32_data = Vec::with_capacity(raw_bytes.len() / 2 * 4);
            for chunk in raw_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f16_val = f16::from_bits(bits);
                let f32_val = f16_val.to_f32();
                f32_data.extend_from_slice(&f32_val.to_le_bytes());
            }
            Ok(f32_data)
        }
        GGUFTensorType::BF16 => {
            let mut f32_data = Vec::with_capacity(raw_bytes.len() / 2 * 4);
            for chunk in raw_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let bf16_val = bf16::from_bits(bits);
                let f32_val = bf16_val.to_f32();
                f32_data.extend_from_slice(&f32_val.to_le_bytes());
            }
            Ok(f32_data)
        }
        GGUFTensorType::Q8_0 => {
            let mut f32_data = Vec::new();
            let chunk_size = 10;
            for chunk in raw_bytes.chunks(chunk_size) {
                if chunk.len() < chunk_size { break; }
                let scale = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                for i in 0..8 {
                    let val = (chunk[2 + i] as i8) as f32 * scale;
                    f32_data.extend_from_slice(&val.to_le_bytes());
                }
            }
            Ok(f32_data)
        }
        GGUFTensorType::Q4_0 => {
            let mut f32_data = Vec::new();
            let chunk_size = 6;
            for chunk in raw_bytes.chunks(chunk_size) {
                if chunk.len() < chunk_size { break; }
                let scale = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                for i in 0..4 {
                    let packed = chunk[2 + i];
                    let v0 = ((packed & 0x0F) as i8) as f32 * scale;
                    let v1 = (((packed >> 4) & 0x0F) as i8) as f32 * scale;
                    f32_data.extend_from_slice(&v0.to_le_bytes());
                    f32_data.extend_from_slice(&v1.to_le_bytes());
                }
            }
            Ok(f32_data)
        }
        GGUFTensorType::Q4_K => {
            let mut f32_data = Vec::new();
            let chunk_size = 6;
            for chunk in raw_bytes.chunks(chunk_size) {
                if chunk.len() < chunk_size { break; }
                let scale = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                for i in 0..4 {
                    let packed = chunk[2 + i];
                    let v0 = ((packed & 0x0F) as i8) as f32 * scale;
                    let v1 = (((packed >> 4) & 0x0F) as i8) as f32 * scale;
                    f32_data.extend_from_slice(&v0.to_le_bytes());
                    f32_data.extend_from_slice(&v1.to_le_bytes());
                }
            }
            Ok(f32_data)
        }
        GGUFTensorType::Q6_K => {
            let mut f32_data = Vec::new();
            let chunk_size = 6;
            for chunk in raw_bytes.chunks(chunk_size) {
                if chunk.len() < chunk_size { break; }
                let scale = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                for i in 0..4 {
                    let packed = chunk[2 + i];
                    let v0 = ((packed & 0x0F) as i8) as f32 * scale;
                    let v1 = (((packed >> 4) & 0x0F) as i8) as f32 * scale;
                    f32_data.extend_from_slice(&v0.to_le_bytes());
                    f32_data.extend_from_slice(&v1.to_le_bytes());
                }
            }
            Ok(f32_data)
        }
        _ => {
            let f32_data: Vec<f32> = raw_bytes.chunks(4)
                .map(|chunk| {
                    if chunk.len() == 4 {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    } else { 0.0 }
                })
                .collect();
            Ok(bytemuck::cast_slice(&f32_data).to_vec())
        }
    }
}

fn map_gguf_type(tensor_type: GGUFTensorType) -> Result<DataType, String> {
    match tensor_type {
        GGUFTensorType::F32 => Ok(DataType::F32),
        GGUFTensorType::F16 => Ok(DataType::F16),
        GGUFTensorType::BF16 => Ok(DataType::BF16),
        GGUFTensorType::I8 => Ok(DataType::I8),
        GGUFTensorType::I32 => Ok(DataType::I32),
        GGUFTensorType::I64 => Ok(DataType::I64),
        GGUFTensorType::Q8_0 => Ok(DataType::I8),
        GGUFTensorType::Q4_0 => Ok(DataType::I8),
        GGUFTensorType::Q4_K => Ok(DataType::I8),
        GGUFTensorType::Q6_K => Ok(DataType::I8),
        _ => Ok(DataType::F32),
    }
}

// ============================================================
// GGUF 导入器 (内部函数)
// ============================================================

pub fn import_gguf_internal(path: &str) -> Result<DagGraph, String> {
    let reader = open_gguf_file(path)
        .map_err(|e| format!("Failed to open GGUF file: {}", e))?;

    let metadata = read_metadata(&reader)?;

    if !SUPPORTED_ARCHITECTURES.contains(&metadata.architecture.as_str()) {
        return Err(format!(
            "Unsupported architecture: {}. Supported: {:?}",
            metadata.architecture, SUPPORTED_ARCHITECTURES
        ));
    }

    let tensor_infos = reader.tensor_infos();
    let mut tensors = Vec::new();

    for tensor_info in tensor_infos {
        let data = extract_tensor_data(&tensor_info)?;
        let shape = tensor_info.shape.dimensions.iter().map(|&x| x as usize).collect();
        let dtype = map_gguf_type(tensor_info.tensor_type)?;

        tensors.push(GGUFTensor {
            name: tensor_info.name.clone(),
            data,
            shape,
            dtype,
            quant_type: tensor_info.tensor_type,
        });
    }

    build_transformer_dag(&metadata, &tensors)
}

// ============================================================
// GGUF 导出器 (内部函数)
// ============================================================

pub fn export_gguf_internal(
    graph: &DagGraph,
    path: &str,
    quant_type: QuantType,
) -> Result<(), String> {
    let tensors = extract_weights_from_graph(graph)?;
    let metadata = build_gguf_metadata(graph)?;

    let mut builder = GGUFBuilder::new();

    for (key, value) in metadata {
        builder = builder.add_metadata(key, value);
    }

    for tensor in tensors {
        let data = quantize_tensor(&tensor.data, quant_type)?;
        let tensor_type = quant_type.to_gguf_type();
        let shape: Vec<u64> = tensor.shape.iter().map(|&x| x as u64).collect();

        builder = builder.add_tensor(&tensor.name, shape, tensor_type, data)
            .map_err(|e| format!("Failed to add tensor {}: {}", tensor.name, e))?;
    }

    builder.build_to_file(path)
        .map_err(|e| format!("Failed to write GGUF file: {}", e))?;

    Ok(())
}

// ============================================================
// 权重提取
// ============================================================

#[derive(Debug, Clone)]
struct ExportTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
}

fn extract_weights_from_graph(graph: &DagGraph) -> Result<Vec<ExportTensor>, String> {
    let mut result = Vec::new();
    for (&id, data) in &graph.constants {
        if let Some(value) = graph.values.get(&id) {
            let shape: Vec<usize> = value.ty.shape.iter()
                .map(|&x| if x == -1 { 0 } else { x as usize })
                .collect();
            result.push(ExportTensor {
                name: value.name.clone(),
                data: data.clone(),
                shape,
            });
        }
    }
    Ok(result)
}

// ============================================================
// 元数据构建
// ============================================================

fn build_gguf_metadata(graph: &DagGraph) -> Result<HashMap<String, MetadataValue>, String> {
    let mut metadata = HashMap::new();
    metadata.insert("general.architecture".to_string(), MetadataValue::String("llama".to_string()));
    metadata.insert("general.name".to_string(), MetadataValue::String(graph.name.clone()));
    metadata.insert("general.version".to_string(), MetadataValue::U64(1));
    metadata.insert("llama.context_length".to_string(), MetadataValue::U64(4096));
    metadata.insert("llama.embedding_length".to_string(), MetadataValue::U64(4096));
    metadata.insert("llama.block_count".to_string(), MetadataValue::U64(32));
    metadata.insert("llama.attention_head_count".to_string(), MetadataValue::U64(32));
    metadata.insert("llama.attention_head_count_kv".to_string(), MetadataValue::U64(8));
    metadata.insert("llama.vocab_size".to_string(), MetadataValue::U64(32000));
    metadata.insert("llama.feed_forward_length".to_string(), MetadataValue::U64(11008));
    metadata.insert("llama.hidden_act".to_string(), MetadataValue::String("silu".to_string()));
    metadata.insert("llama.attention_norm_eps".to_string(), MetadataValue::F32(1e-5));
    metadata.insert("llama.rope_theta".to_string(), MetadataValue::F32(10000.0));
    Ok(metadata)
}

// ============================================================
// 量化
// ============================================================

fn quantize_tensor(data: &[u8], quant_type: QuantType) -> Result<Vec<u8>, String> {
    if quant_type == QuantType::F32 {
        return Ok(data.to_vec());
    }

    let f32_data: Vec<f32> = data.chunks(4)
        .map(|chunk| {
            if chunk.len() == 4 {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            } else { 0.0 }
        })
        .collect();

    match quant_type {
        QuantType::F16 => {
            let f16_data: Vec<u16> = f32_data.iter()
                .map(|&x| f16::from_f32(x).to_bits())
                .collect();
            Ok(bytemuck::cast_slice(&f16_data).to_vec())
        }
        QuantType::BF16 => {
            let bf16_data: Vec<u16> = f32_data.iter()
                .map(|&x| bf16::from_f32(x).to_bits())
                .collect();
            Ok(bytemuck::cast_slice(&bf16_data).to_vec())
        }
        QuantType::Q8_0 => {
            let mut result = Vec::new();
            for chunk in f32_data.chunks(8) {
                let max_val = chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };
                let scale_f16 = f16::from_f32(scale);
                result.extend_from_slice(&scale_f16.to_le_bytes());
                for &x in chunk {
                    let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
                    result.push(q as u8);
                }
            }
            Ok(result)
        }
        QuantType::Q4_0 => {
            let mut result = Vec::new();
            for chunk in f32_data.chunks(16) {
                let max_val = chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let scale = if max_val > 0.0 { max_val / 7.0 } else { 1.0 };
                let scale_f16 = f16::from_f32(scale);
                result.extend_from_slice(&scale_f16.to_le_bytes());
                for pair in chunk.chunks(2) {
                    let v0 = ((pair[0] / scale).round().clamp(-7.0, 7.0) as i8 & 0x0F) as u8;
                    let v1 = ((pair[1] / scale).round().clamp(-7.0, 7.0) as i8 & 0x0F) as u8;
                    result.push(v0 | (v1 << 4));
                }
            }
            Ok(result)
        }
        QuantType::Q4_K => {
            let mut result = Vec::new();
            for chunk in f32_data.chunks(16) {
                let max_val = chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let scale = if max_val > 0.0 { max_val / 7.0 } else { 1.0 };
                let scale_f16 = f16::from_f32(scale);
                result.extend_from_slice(&scale_f16.to_le_bytes());
                for pair in chunk.chunks(2) {
                    let v0 = ((pair[0] / scale).round().clamp(-7.0, 7.0) as i8 & 0x0F) as u8;
                    let v1 = ((pair[1] / scale).round().clamp(-7.0, 7.0) as i8 & 0x0F) as u8;
                    result.push(v0 | (v1 << 4));
                }
            }
            Ok(result)
        }
        _ => Ok(data.to_vec()),
    }
}

// ============================================================
// Transformer DAG 构建器
// ============================================================

fn get_weight(weight_map: &HashMap<String, u64>, name: &str) -> Result<u64, String> {
    if let Some(&id) = weight_map.get(name) {
        return Ok(id);
    }

    // 尝试模糊匹配
    for key in weight_map.keys() {
        if key.contains(name) || name.contains(key) {
            if let Some(&id) = weight_map.get(key) {
                return Ok(id);
            }
        }
    }

    Err(format!("Weight not found: {}", name))
}

fn matmul_op(
    dag: &mut DagGraph,
    name: &str,
    input: u64,
    weight: u64,
    op_counter: &mut u64,
) -> Result<u64, String> {
    let output = dag.add_value(name, DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, 0],
    });
    let op = Op {
        id: *op_counter,
        name: format!("matmul_{}", name),
        op_type: "matmul".to_string(),
        inputs: vec![input, weight],
        outputs: vec![output],
        attrs: HashMap::new(),
    };
    dag.ops.insert(*op_counter, op);
    *op_counter += 1;
    Ok(output)
}

fn build_layer(
    dag: &mut DagGraph,
    weight_map: &HashMap<String, u64>,
    metadata: &GGUFModelMetadata,
    layer: usize,
    input: u64,
    mut op_counter: u64,
) -> Result<(u64, u64), String> {
    let prefix = format!("layers.{}.", layer);

    let q_proj = get_weight(weight_map, &format!("{}attention.q_proj.weight", prefix))?;
    let k_proj = get_weight(weight_map, &format!("{}attention.k_proj.weight", prefix))?;
    let v_proj = get_weight(weight_map, &format!("{}attention.v_proj.weight", prefix))?;
    let o_proj = get_weight(weight_map, &format!("{}attention.o_proj.weight", prefix))?;
    let gate_proj = get_weight(weight_map, &format!("{}mlp.gate_proj.weight", prefix))?;
    let up_proj = get_weight(weight_map, &format!("{}mlp.up_proj.weight", prefix))?;
    let down_proj = get_weight(weight_map, &format!("{}mlp.down_proj.weight", prefix))?;
    let attn_norm = get_weight(weight_map, &format!("{}attention_norm.weight", prefix))?;
    let mlp_norm = get_weight(weight_map, &format!("{}mlp_norm.weight", prefix))?;

    // RMSNorm
    let attn_norm_out = dag.add_value(&format!("attn_norm_{}", layer), DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.hidden_size as i64],
    });
    let attn_norm_op = Op {
        id: op_counter,
        name: format!("attn_norm_{}", layer),
        op_type: "rms_norm".to_string(),
        inputs: vec![input, attn_norm],
        outputs: vec![attn_norm_out],
        attrs: {
            let mut attrs = HashMap::new();
            attrs.insert("eps".to_string(), AttrValue::Float(metadata.rms_norm_eps as f64));
            attrs
        },
    };
    dag.ops.insert(op_counter, attn_norm_op);
    op_counter += 1;

    // Q, K, V
    let q = matmul_op(dag, &format!("q_{}", layer), attn_norm_out, q_proj, &mut op_counter)?;
    let k = matmul_op(dag, &format!("k_{}", layer), attn_norm_out, k_proj, &mut op_counter)?;
    let v = matmul_op(dag, &format!("v_{}", layer), attn_norm_out, v_proj, &mut op_counter)?;

    // SDPA
    let attn_out = dag.add_value(&format!("attn_{}", layer), DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.hidden_size as i64],
    });
    let sdpa_op = Op {
        id: op_counter,
        name: format!("sdpa_{}", layer),
        op_type: "scaled_dot_product_attention".to_string(),
        inputs: vec![q, k, v],
        outputs: vec![attn_out],
        attrs: {
            let mut attrs = HashMap::new();
            attrs.insert("scale".to_string(), AttrValue::Float(0.0));
            attrs.insert("is_causal".to_string(), AttrValue::Bool(true));
            attrs
        },
    };
    dag.ops.insert(op_counter, sdpa_op);
    op_counter += 1;

    // O
    let o = matmul_op(dag, &format!("o_{}", layer), attn_out, o_proj, &mut op_counter)?;

    // Residual 1
    let res1 = dag.add_value(&format!("res1_{}", layer), DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.hidden_size as i64],
    });
    let add1_op = Op {
        id: op_counter,
        name: format!("add1_{}", layer),
        op_type: "add".to_string(),
        inputs: vec![input, o],
        outputs: vec![res1],
        attrs: HashMap::new(),
    };
    dag.ops.insert(op_counter, add1_op);
    op_counter += 1;

    // MLP RMSNorm
    let mlp_norm_out = dag.add_value(&format!("mlp_norm_{}", layer), DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.hidden_size as i64],
    });
    let mlp_norm_op = Op {
        id: op_counter,
        name: format!("mlp_norm_{}", layer),
        op_type: "rms_norm".to_string(),
        inputs: vec![res1, mlp_norm],
        outputs: vec![mlp_norm_out],
        attrs: {
            let mut attrs = HashMap::new();
            attrs.insert("eps".to_string(), AttrValue::Float(metadata.rms_norm_eps as f64));
            attrs
        },
    };
    dag.ops.insert(op_counter, mlp_norm_op);
    op_counter += 1;

    // Gate (SiLU)
    let gate = matmul_op(dag, &format!("gate_{}", layer), mlp_norm_out, gate_proj, &mut op_counter)?;
    let gate_act = dag.add_value(&format!("gate_act_{}", layer), DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.intermediate_size as i64],
    });
    let silu_op = Op {
        id: op_counter,
        name: format!("silu_{}", layer),
        op_type: "silu".to_string(),
        inputs: vec![gate],
        outputs: vec![gate_act],
        attrs: HashMap::new(),
    };
    dag.ops.insert(op_counter, silu_op);
    op_counter += 1;

    // Up
    let up = matmul_op(dag, &format!("up_{}", layer), mlp_norm_out, up_proj, &mut op_counter)?;

    // Up * Gate_Act
    let mlp = dag.add_value(&format!("mlp_{}", layer), DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.intermediate_size as i64],
    });
    let mul_op = Op {
        id: op_counter,
        name: format!("mul_{}", layer),
        op_type: "mul".to_string(),
        inputs: vec![up, gate_act],
        outputs: vec![mlp],
        attrs: HashMap::new(),
    };
    dag.ops.insert(op_counter, mul_op);
    op_counter += 1;

    // Down
    let down = matmul_op(dag, &format!("down_{}", layer), mlp, down_proj, &mut op_counter)?;

    // Residual 2
    let res2 = dag.add_value(&format!("res2_{}", layer), DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.hidden_size as i64],
    });
    let add2_op = Op {
        id: op_counter,
        name: format!("add2_{}", layer),
        op_type: "add".to_string(),
        inputs: vec![res1, down],
        outputs: vec![res2],
        attrs: HashMap::new(),
    };
    dag.ops.insert(op_counter, add2_op);
    op_counter += 1;

    Ok((res2, op_counter))
}

fn build_transformer_dag(
    metadata: &GGUFModelMetadata,
    tensors: &[GGUFTensor],
) -> Result<DagGraph, String> {
    let mut dag = DagGraph::new(&format!("{}_model", metadata.architecture));

    let mut weight_map: HashMap<String, u64> = HashMap::new();
    let mut op_counter: u64 = 0;

    for tensor in tensors {
        let shape: Vec<i64> = tensor.shape.iter().map(|&x| x as i64).collect();
        let ty = DagTensorType {
            dtype: tensor.dtype,
            shape: shape.clone(),
        };
        let id = dag.add_constant(&tensor.name, ty, tensor.data.clone());
        weight_map.insert(tensor.name.clone(), id);
    }

    // 输入
    let input_id = dag.add_value("input", DagTensorType {
        dtype: DataType::I64,
        shape: vec![1, -1],
    });
    dag.inputs.push(input_id);

    // Token Embedding
    let embed_weight = get_weight(&weight_map, "token_embedding.weight")?;
    let embed_output = dag.add_value("embed_output", DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.hidden_size as i64],
    });
    let embed_op = Op {
        id: op_counter,
        name: "embedding".to_string(),
        op_type: "embedding".to_string(),
        inputs: vec![input_id, embed_weight],
        outputs: vec![embed_output],
        attrs: HashMap::new(),
    };
    dag.ops.insert(op_counter, embed_op);
    op_counter += 1;

    // 各层
    let mut current = embed_output;
    for layer in 0..metadata.num_layers {
        let (next, count) = build_layer(&mut dag, &weight_map, metadata, layer, current, op_counter)?;
        current = next;
        op_counter = count;
    }

    // Final RMSNorm
    let norm_weight = get_weight(&weight_map, "final_norm.weight")?;
    let norm_output = dag.add_value("final_norm_output", DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.hidden_size as i64],
    });
    let norm_op = Op {
        id: op_counter,
        name: "final_rms_norm".to_string(),
        op_type: "rms_norm".to_string(),
        inputs: vec![current, norm_weight],
        outputs: vec![norm_output],
        attrs: {
            let mut attrs = HashMap::new();
            attrs.insert("eps".to_string(), AttrValue::Float(metadata.rms_norm_eps as f64));
            attrs
        },
    };
    dag.ops.insert(op_counter, norm_op);
    op_counter += 1;
    current = norm_output;

    // LM Head
    let lm_head = get_weight(&weight_map, "lm_head.weight")?;
    let output = dag.add_value("output", DagTensorType {
        dtype: DataType::F32,
        shape: vec![1, -1, metadata.vocab_size as i64],
    });
    let lm_op = Op {
        id: op_counter,
        name: "lm_head".to_string(),
        op_type: "matmul".to_string(),
        inputs: vec![current, lm_head],
        outputs: vec![output],
        attrs: HashMap::new(),
    };
    dag.ops.insert(op_counter, lm_op);
    op_counter += 1;

    dag.outputs = vec![output];
    Ok(dag)
}

// ============================================================
// PyO3 绑定 (暴露给 Python)
// ============================================================

#[pyfunction]
pub fn import_gguf(path: &str) -> PyResult<Py<PyAny>> {
    let dag = import_gguf_internal(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Python::with_gil(|py| {
        let py_dag = crate::ir::serialize::PyDagGraph { inner: dag };
        Ok(Py::new(py, py_dag)?.into_any())
    })
}

#[pyfunction]
pub fn export_gguf(path: &str, dag: Py<PyAny>, quant_type: &str) -> PyResult<()> {
    Python::with_gil(|py| {
        let py_dag = dag.extract::<crate::ir::serialize::PyDagGraph>(py)?;
        let qt = QuantType::from_str(quant_type)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid quant type"))?;

        export_gguf_internal(&py_dag.inner, path, qt)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    })
}
