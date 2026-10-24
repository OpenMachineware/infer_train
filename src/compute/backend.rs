// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Backend trait for unified execution across devices.

use std::sync::Arc;

use super::graph::{ComputeGraph, OpType};
use super::storage::{Device, Storage};
use crate::model::tensor::DataType;

/// Execution context for a backend.
pub struct ExecutionContext {
    /// Device being used.
    pub device: Device,
    /// KV cache position (for attention).
    pub seq_len: usize,
    /// Batch size.
    pub batch_size: usize,
}

/// Backend trait for tensor operations.
/// Note: No Send/Sync requirements because MetalContext uses RefCell internally.
pub trait Backend {
    /// Get the device type.
    fn device(&self) -> Device;

    /// Allocate a tensor on this device.
    fn alloc_tensor(&self, shape: &[usize], dtype: DataType) -> Storage;

    /// Allocate a tensor with initial data.
    fn alloc_tensor_with_data(
        &self,
        shape: &[usize],
        dtype: DataType,
        data: &[u8],
    ) -> Storage;

    /// Execute a compute graph.
    /// Returns output tensors by name.
    fn execute(
        &self,
        graph: &ComputeGraph,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Storage>, String>;

    /// Copy tensor data to CPU.
    fn copy_to_cpu(&self, storage: &Storage) -> Result<Vec<u8>, String>;

    /// Copy tensor data from CPU.
    fn copy_from_cpu(
        &self,
        data: &[u8],
        storage: &mut Storage,
    ) -> Result<(), String>;

    // Primitive operations (for individual op execution)

    /// RMS normalization.
    fn rms_norm(
        &self,
        input: &Storage,
        weight: &Storage,
        output: &mut Storage,
        eps: f32,
    ) -> Result<(), String>;

    /// Matrix multiplication: output = input @ weight.
    /// Weight is quantized, input is F32, output is F32.
    fn matmul(
        &self,
        input: &Storage,
        weight: &Storage,
        weight_dtype: DataType,
        output: &mut Storage,
    ) -> Result<(), String>;

    /// Flash attention.
    fn flash_attn(
        &self,
        query: &Storage,
        k_cache: &Storage,
        v_cache: &Storage,
        output: &mut Storage,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        seq_len: usize,
    ) -> Result<(), String>;

    /// RoPE position encoding.
    fn rope(
        &self,
        input: &Storage,
        pos: usize,
        output: &mut Storage,
        n_heads: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> Result<(), String>;

    /// Element-wise add.
    fn add(
        &self,
        a: &Storage,
        b: &Storage,
        output: &mut Storage,
    ) -> Result<(), String>;

    /// Element-wise multiply.
    fn mul(
        &self,
        a: &Storage,
        b: &Storage,
        output: &mut Storage,
    ) -> Result<(), String>;

    /// SiLU activation.
    fn silu(&self, input: &Storage, output: &mut Storage)
    -> Result<(), String>;
}

/// CPU backend implementation.
pub struct CpuBackend {
    #[allow(unused)]
    num_threads: usize,
}

impl CpuBackend {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn alloc_tensor(&self, shape: &[usize], dtype: DataType) -> Storage {
        let num_elems: usize = shape.iter().product();
        let size = num_elems * dtype.block_size() / dtype.elems_per_block();
        Storage::cpu(vec![0u8; size])
    }

    fn alloc_tensor_with_data(
        &self,
        _shape: &[usize],
        _dtype: DataType,
        data: &[u8],
    ) -> Storage {
        Storage::cpu(data.to_vec())
    }

    fn execute(
        &self,
        graph: &ComputeGraph,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Storage>, String> {
        // For now, execute ops one by one
        // TODO: Parallelize independent ops
        let mut tensor_storage: std::collections::HashMap<String, Storage> =
            std::collections::HashMap::new();

        // Initialize input tensors
        for input_name in &graph.inputs {
            if let Some(tensor) = graph.get_tensor(input_name) {
                if let Some(storage) = &tensor.storage {
                    tensor_storage.insert(input_name.clone(), storage.clone());
                }
            }
        }

        // Initialize weight tensors
        for (name, tensor) in &graph.tensors {
            if let Some(storage) = &tensor.storage {
                tensor_storage.insert(name.clone(), storage.clone());
            }
        }

        // Execute each operation
        for node in &graph.nodes {
            // Get inputs
            let inputs: Vec<&Storage> = node
                .inputs
                .iter()
                .map(|name| {
                    tensor_storage
                        .get(name)
                        .ok_or_else(|| format!("Tensor '{}' not found", name))
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Allocate output
            let output_shape = Self::infer_output_shape(&node.op, &inputs)?;
            let output_dtype = Self::infer_output_dtype(&node.op, &inputs);
            let mut output_storage =
                self.alloc_tensor(&output_shape, output_dtype);

            // Execute operation
            match &node.op {
                OpType::RmsNorm { eps } => {
                    self.rms_norm(
                        inputs[0],
                        inputs[1],
                        &mut output_storage,
                        *eps,
                    )?;
                }
                OpType::MatMul => {
                    // Infer weight dtype from second input
                    let weight_dtype = DataType::F32; // TODO: Get from tensor metadata
                    self.matmul(
                        inputs[0],
                        inputs[1],
                        weight_dtype,
                        &mut output_storage,
                    )?;
                }
                OpType::FlashAttn { n_heads, head_dim, n_kv_heads, .. } => {
                    self.flash_attn(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        &mut output_storage,
                        *n_heads,
                        *head_dim,
                        *n_kv_heads,
                        ctx.seq_len,
                    )?;
                }
                OpType::Add => {
                    self.add(inputs[0], inputs[1], &mut output_storage)?;
                }
                OpType::Mul => {
                    self.mul(inputs[0], inputs[1], &mut output_storage)?;
                }
                OpType::Silu => {
                    self.silu(inputs[0], &mut output_storage)?;
                }
                _ => return Err(format!("Unsupported op: {:?}", node.op)),
            }

            tensor_storage.insert(node.output.clone(), output_storage);
        }

        // Collect outputs
        let outputs: Vec<Storage> = graph
            .outputs
            .iter()
            .map(|name| {
                tensor_storage
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("Output '{}' not found", name))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(outputs)
    }

    fn copy_to_cpu(&self, storage: &Storage) -> Result<Vec<u8>, String> {
        match storage {
            Storage::Cpu(data) => Ok(data.as_ref().to_vec()),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Storage::Metal(_) => {
                Err("Cannot copy Metal buffer to CPU on CPU backend"
                    .to_string())
            }
        }
    }

    fn copy_from_cpu(
        &self,
        data: &[u8],
        storage: &mut Storage,
    ) -> Result<(), String> {
        match storage {
            Storage::Cpu(buf) => {
                Arc::make_mut(buf).copy_from_slice(data);
                Ok(())
            }
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Storage::Metal(_) => {
                Err("Cannot copy to Metal buffer on CPU backend".to_string())
            }
        }
    }

    fn rms_norm(
        &self,
        input: &Storage,
        weight: &Storage,
        output: &mut Storage,
        _eps: f32,
    ) -> Result<(), String> {
        let _input_data = input.as_bytes().ok_or("Input not on CPU")?;
        let _weight_data = weight.as_bytes().ok_or("Weight not on CPU")?;
        let _output_data = output.as_bytes().ok_or("Output not on CPU")?;

        // TODO: Implement RMS norm
        Err("RMS norm not implemented on CPU backend".to_string())
    }

    fn matmul(
        &self,
        _input: &Storage,
        _weight: &Storage,
        _weight_dtype: DataType,
        _output: &mut Storage,
    ) -> Result<(), String> {
        // Use existing CPU GEMV/GEMM implementation
        Err("MatMul not implemented on CPU backend".to_string())
    }

    fn flash_attn(
        &self,
        _query: &Storage,
        _k_cache: &Storage,
        _v_cache: &Storage,
        _output: &mut Storage,
        _n_heads: usize,
        _head_dim: usize,
        _n_kv_heads: usize,
        _seq_len: usize,
    ) -> Result<(), String> {
        // Use existing CPU attention implementation
        Err("Flash attention not implemented on CPU backend".to_string())
    }

    fn rope(
        &self,
        _input: &Storage,
        _pos: usize,
        _output: &mut Storage,
        _n_heads: usize,
        _head_dim: usize,
        _freq_base: f32,
    ) -> Result<(), String> {
        Err("RoPE not implemented on CPU backend".to_string())
    }

    fn add(
        &self,
        _a: &Storage,
        _b: &Storage,
        _output: &mut Storage,
    ) -> Result<(), String> {
        Err("Add not implemented on CPU backend".to_string())
    }

    fn mul(
        &self,
        _a: &Storage,
        _b: &Storage,
        _output: &mut Storage,
    ) -> Result<(), String> {
        Err("Mul not implemented on CPU backend".to_string())
    }

    fn silu(
        &self,
        _input: &Storage,
        _output: &mut Storage,
    ) -> Result<(), String> {
        Err("SiLU not implemented on CPU backend".to_string())
    }
}

impl CpuBackend {
    fn infer_output_shape(
        op: &OpType,
        _inputs: &[&Storage],
    ) -> Result<Vec<usize>, String> {
        match op {
            OpType::RmsNorm { .. } => {
                // Output has same shape as input
                // TODO: Infer from input
                Ok(vec![])
            }
            OpType::MatMul => {
                // Output shape: (batch, out_features)
                Ok(vec![])
            }
            OpType::FlashAttn { n_heads, head_dim, .. } => {
                Ok(vec![*n_heads * head_dim])
            }
            OpType::Add | OpType::Mul => {
                // Same shape as inputs
                Ok(vec![])
            }
            OpType::Silu => Ok(vec![]),
            _ => Ok(vec![]),
        }
    }

    fn infer_output_dtype(op: &OpType, _inputs: &[&Storage]) -> DataType {
        match op {
            OpType::MatMul => DataType::F32,
            OpType::FlashAttn { .. } => DataType::F32,
            OpType::RmsNorm { .. } => DataType::F32,
            OpType::Add | OpType::Mul => DataType::F32,
            OpType::Silu => DataType::F32,
            _ => DataType::F32,
        }
    }
}

// Note: MetalBackend is in a separate file due to feature gating
