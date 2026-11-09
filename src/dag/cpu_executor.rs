// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// CPU DAG Executor - Execute DAG operations on CPU with zero allocation
//
// This executor reuses the existing DAG infrastructure for CPU path,
// providing graph plan optimization benefits to CPU inference.

use super::graph::{ComputeDag, Operation, OptimizedDag};
use crate::inference::KVCache;
use crate::model::LlamaModel;
use std::collections::HashMap;

/// CPU tensor: pre-allocated buffer with shape tracking
#[derive(Clone)]
pub struct CpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>, // Tensor shape, e.g., [hidden_size, n_tokens]
}

impl CpuTensor {
    pub fn new(size: usize) -> Self {
        CpuTensor { data: vec![0.0f32; size], shape: vec![size] }
    }

    pub fn with_shape(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        CpuTensor { data: vec![0.0f32; size], shape }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn nelements(&self) -> usize {
        self.data.len()
    }

    pub fn is_batched(&self) -> bool {
        self.shape.len() > 1 && self.shape.last().copied().unwrap_or(1) > 1
    }

    pub fn batch_size(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }
}

/// CPU Tensor Registry: maps tensor IDs to CPU buffers
pub struct CpuTensorRegistry {
    tensors: Vec<Option<CpuTensor>>,
    max_tensor_id: usize,
}

impl CpuTensorRegistry {
    pub fn new() -> Self {
        Self { tensors: Vec::new(), max_tensor_id: 0 }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut tensors = Vec::with_capacity(capacity);
        tensors.resize(capacity, None);
        Self { tensors, max_tensor_id: 0 }
    }

    pub fn get(&self, id: usize) -> Option<&CpuTensor> {
        self.tensors.get(id).and_then(|t| t.as_ref())
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut CpuTensor> {
        self.tensors.get_mut(id).and_then(|t| t.as_mut())
    }

    pub fn get_slice(&self, id: usize) -> Option<&[f32]> {
        self.get(id).map(|t| t.as_slice())
    }

    pub fn insert(&mut self, id: usize, tensor: CpuTensor) {
        if id >= self.tensors.len() {
            let new_len = (id + 1).next_power_of_two();
            self.tensors.resize(new_len, None);
        }
        self.tensors[id] = Some(tensor);
        if id > self.max_tensor_id {
            self.max_tensor_id = id;
        }
    }

    /// Create or resize a tensor (1D convenience method)
    pub fn ensure_tensor(&mut self, id: usize, size: usize) {
        if id >= self.tensors.len() {
            let new_len = (id + 1).next_power_of_two();
            self.tensors.resize(new_len, None);
        }

        match &mut self.tensors[id] {
            Some(tensor) => {
                if tensor.data.len() < size {
                    tensor.data.resize(size, 0.0);
                    tensor.shape = vec![size];
                }
            }
            None => {
                self.tensors[id] = Some(CpuTensor::new(size));
            }
        }
    }

    /// Create or resize a tensor with given shape
    pub fn ensure_tensor_with_shape(&mut self, id: usize, shape: Vec<usize>) {
        let size: usize = shape.iter().product();
        if id >= self.tensors.len() {
            let new_len = (id + 1).next_power_of_two();
            self.tensors.resize(new_len, None);
        }

        match &mut self.tensors[id] {
            Some(tensor) => {
                if tensor.data.len() < size {
                    tensor.data.resize(size, 0.0);
                    tensor.shape = shape;
                }
            }
            None => {
                self.tensors[id] = Some(CpuTensor::with_shape(shape));
            }
        }
    }
}

/// CPU Execution Context
pub struct CpuExecutionContext<'a> {
    pub model: &'a LlamaModel,
    pub kv_cache: &'a mut KVCache,
    pub tensor_registry: CpuTensorRegistry,
    pub weight_bindings: &'a HashMap<usize, crate::model::Tensor>,
    pub layer_idx: usize,
    pub position: usize,
    /// Track K tensor IDs per layer (from KVCacheUpdate operations)
    pub k_tensor_ids: HashMap<usize, usize>,
    /// Track V tensor IDs per layer (from KVCacheUpdate operations)
    pub v_tensor_ids: HashMap<usize, usize>,
    /// Work buffers for batch operations (optional, used for prefill)
    pub buffers: Option<&'a mut crate::cpu::CpuWorkBuffers>,
}

impl<'a> CpuExecutionContext<'a> {
    pub fn new(
        model: &'a LlamaModel,
        kv_cache: &'a mut KVCache,
        weight_bindings: &'a HashMap<usize, crate::model::Tensor>,
    ) -> Self {
        Self {
            model,
            kv_cache,
            tensor_registry: CpuTensorRegistry::new(),
            weight_bindings,
            layer_idx: 0,
            position: 0,
            k_tensor_ids: HashMap::new(),
            v_tensor_ids: HashMap::new(),
            buffers: None,
        }
    }

    /// Create with buffers for batch operations
    pub fn with_buffers(
        model: &'a LlamaModel,
        kv_cache: &'a mut KVCache,
        weight_bindings: &'a HashMap<usize, crate::model::Tensor>,
        buffers: &'a mut crate::cpu::CpuWorkBuffers,
    ) -> Self {
        Self {
            model,
            kv_cache,
            tensor_registry: CpuTensorRegistry::new(),
            weight_bindings,
            layer_idx: 0,
            position: 0,
            k_tensor_ids: HashMap::new(),
            v_tensor_ids: HashMap::new(),
            buffers: Some(buffers),
        }
    }
}

/// Execute a single operation on CPU
pub fn execute_operation(
    ctx: &mut CpuExecutionContext,
    op: &Operation,
) -> Option<()> {
    let op_name = match op {
        Operation::EmbeddingLookup { .. } => "EmbeddingLookup",
        Operation::RmsNorm { .. } => "RmsNorm",
        Operation::Gemv { .. } => "Gemv",
        Operation::RoPE { .. } => "RoPE",
        Operation::FlashAttention { .. } => "FlashAttention",
        Operation::KVCacheUpdate { .. } => "KVCacheUpdate",
        Operation::SiLU { .. } => "SiLU",
        Operation::Mul { .. } => "Mul",
        Operation::Add { .. } => "Add",
        Operation::FusedQKVWeight { .. } => "FusedQKVWeight",
        Operation::NoOp => "NoOp",
        _ => "Other",
    };

    match op {
        Operation::EmbeddingLookup {
            token_id,
            embedding_weight: _,
            output,
        } => {
            // Lookup embedding
            let embedding = ctx.model.embedding_lookup_impl(*token_id)?;

            // Store result
            ctx.tensor_registry.ensure_tensor(*output, embedding.len());
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;
            out_tensor.as_mut_slice().copy_from_slice(&embedding);

            Some(())
        }

        Operation::RmsNorm { input, weight, output, epsilon: _ } => {
            // Read input data (copy needed to avoid borrow conflict with output)
            let input_data = ctx.tensor_registry.get_slice(*input)?.to_vec();
            let weight_tensor = ctx.weight_bindings.get(weight)?;

            // Allocate output
            ctx.tensor_registry.ensure_tensor(*output, input_data.len());
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;

            // Execute RMS norm
            ctx.model.apply_rms_norm_into(
                &input_data,
                weight_tensor,
                out_tensor.as_mut_slice(),
            )?;

            Some(())
        }

        Operation::Gemv { input, weight, output, out_size, is_transposed } => {
            // Check if input is batched (prefill) or single (decode)
            let input_tensor = ctx.tensor_registry.get(*input)?;
            let input_data = input_tensor.as_slice().to_vec();
            let batch_size = input_tensor.batch_size();

            let weight_tensor = ctx.weight_bindings.get(weight)?;

            if batch_size > 1 {
                // Batched GEMM for prefill
                // Input shape: [hidden_size, batch_size] in row-major
                // Weight shape: [out_size, hidden_size] (standard) or [hidden_size, out_size] (transposed)

                let hidden_size = input_data.len() / batch_size;

                // Allocate output [out_size, batch_size]
                ctx.tensor_registry.ensure_tensor_with_shape(
                    *output,
                    vec![*out_size, batch_size],
                );
                let out_tensor = ctx.tensor_registry.get_mut(*output)?;
                let out_slice = out_tensor.as_mut_slice();

                // Split input into per-token vectors
                let input_vecs: Vec<&[f32]> = (0..batch_size)
                    .map(|i| {
                        &input_data[i * hidden_size..(i + 1) * hidden_size]
                    })
                    .collect();

                // Use batch GEMM (need buffers)
                if let Some(buffers) = &mut ctx.buffers {
                    let success = ctx.model.gemm_batch_prefill(
                        &input_vecs,
                        weight_tensor,
                        out_slice,
                        &mut buffers.q8k_input,
                        &mut buffers.x_padded,
                        &mut buffers.x_padded_threads,
                        None,
                    );
                    if success { Some(()) } else { None }
                } else {
                    // No buffers available - can't do batch prefill
                    eprintln!(
                        "Error: Batch prefill requires buffers but none available"
                    );
                    None
                }
            } else {
                // Single GEMV for decode
                if *is_transposed {
                    // Transposed GEMV: weight is stored as [in_size, out_size]
                    let result = ctx.model.gemv_transposed(
                        &input_data,
                        weight_tensor,
                        *out_size,
                    )?;
                    ctx.tensor_registry.ensure_tensor(*output, result.len());
                    let out_tensor = ctx.tensor_registry.get_mut(*output)?;
                    out_tensor.as_mut_slice().copy_from_slice(&result);
                } else {
                    // Standard GEMV: weight is stored as [out_size, in_size]
                    ctx.tensor_registry.ensure_tensor(*output, *out_size);
                    let out_tensor = ctx.tensor_registry.get_mut(*output)?;
                    ctx.model.gemv_into(
                        &input_data,
                        weight_tensor,
                        out_tensor.as_mut_slice(),
                    );
                }
                Some(())
            }
        }

        Operation::RoPE { input, output, pos, n_heads, head_dim } => {
            // Check if input is batched
            let input_tensor = ctx.tensor_registry.get(*input)?;
            let input_data = input_tensor.as_slice().to_vec();
            let batch_size = input_tensor.batch_size();

            // Allocate output with same shape
            ctx.tensor_registry
                .ensure_tensor_with_shape(*output, input_tensor.shape.clone());
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;

            if batch_size > 1 {
                // Batched RoPE for prefill
                // Input shape: [n_heads * head_dim, batch_size]
                // Apply RoPE with positions: pos, pos+1, ..., pos+batch_size-1
                let per_token_size = *n_heads * *head_dim;

                for i in 0..batch_size {
                    let start = i * per_token_size;
                    let end = start + per_token_size;
                    let input_slice = &input_data[start..end];
                    let output_slice =
                        &mut out_tensor.as_mut_slice()[start..end];

                    ctx.model.apply_rope_into(
                        input_slice,
                        *pos + i,
                        *n_heads,
                        *head_dim,
                        output_slice,
                    )?;
                }
            } else {
                // Single RoPE for decode
                ctx.model.apply_rope_into(
                    &input_data,
                    *pos,
                    *n_heads,
                    *head_dim,
                    out_tensor.as_mut_slice(),
                )?;
            }

            Some(())
        }

        Operation::FlashAttention {
            q,
            output,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
        } => {
            // Get K and V tensor IDs from context (tracked by KVCacheUpdate)
            let k_tensor_id = ctx.k_tensor_ids.get(layer_idx).copied()?;
            let v_tensor_id = ctx.v_tensor_ids.get(layer_idx).copied()?;

            // Execute attention directly with tensor references
            let q_data = ctx.tensor_registry.get_slice(*q)?;
            let k_data = ctx.tensor_registry.get_slice(k_tensor_id)?;
            let v_data = ctx.tensor_registry.get_slice(v_tensor_id)?;

            let attn_out = ctx.model.attention(
                q_data,
                k_data,
                v_data,
                *layer_idx,
                ctx.kv_cache,
                *n_heads,
                *head_dim,
                *n_kv_heads,
            )?;

            // Store result
            ctx.tensor_registry.ensure_tensor(*output, attn_out.len());
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;
            out_tensor.as_mut_slice().copy_from_slice(&attn_out);

            Some(())
        }

        Operation::KVCacheUpdate { input, layer_idx, is_key, .. } => {
            // Track the tensor ID for K or V so FlashAttention can use it
            if *is_key {
                ctx.k_tensor_ids.insert(*layer_idx, *input);
            } else {
                ctx.v_tensor_ids.insert(*layer_idx, *input);
            }

            // Note: The actual KV cache update is done by the attention function
            // This operation just tracks the tensor ID for later use

            Some(())
        }

        Operation::SiLU { input, output } => {
            // Read input data (copy needed to avoid borrow conflict)
            let input_data = ctx.tensor_registry.get_slice(*input)?.to_vec();
            let len = input_data.len();

            // Allocate output
            ctx.tensor_registry.ensure_tensor(*output, len);
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;

            // Execute SiLU using std::thread::scope
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::thread;

            let n_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            let counter = AtomicUsize::new(0);
            let chunk_size = 256;
            let out_addr = out_tensor.as_mut_slice().as_mut_ptr() as usize;

            thread::scope(|s| {
                for _ in 0..n_threads {
                    s.spawn(|| {
                        let out_ptr = out_addr as *mut f32;
                        loop {
                            let chunk_start = counter
                                .fetch_add(chunk_size, Ordering::Relaxed);
                            if chunk_start >= len {
                                break;
                            }
                            let chunk_end = (chunk_start + chunk_size).min(len);
                            for i in chunk_start..chunk_end {
                                let x = input_data[i];
                                unsafe {
                                    *out_ptr.add(i) = x / (1.0 + (-x).exp());
                                }
                            }
                        }
                    });
                }
            });

            Some(())
        }

        Operation::Mul { lhs, rhs, output } => {
            // Read input data (copies needed to avoid borrow conflicts)
            let lhs_data = ctx.tensor_registry.get_slice(*lhs)?.to_vec();
            let rhs_data = ctx.tensor_registry.get_slice(*rhs)?.to_vec();
            let len = lhs_data.len();

            // Allocate output
            ctx.tensor_registry.ensure_tensor(*output, len);
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;

            // Execute element-wise multiply using std::thread::scope
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::thread;

            let n_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            let counter = AtomicUsize::new(0);
            let chunk_size = 256;
            let out_addr = out_tensor.as_mut_slice().as_mut_ptr() as usize;

            thread::scope(|s| {
                for _ in 0..n_threads {
                    s.spawn(|| {
                        let out_ptr = out_addr as *mut f32;
                        loop {
                            let chunk_start = counter
                                .fetch_add(chunk_size, Ordering::Relaxed);
                            if chunk_start >= len {
                                break;
                            }
                            let chunk_end = (chunk_start + chunk_size).min(len);
                            for i in chunk_start..chunk_end {
                                unsafe {
                                    *out_ptr.add(i) = lhs_data[i] * rhs_data[i];
                                }
                            }
                        }
                    });
                }
            });

            Some(())
        }

        Operation::Add { lhs, rhs, output } => {
            // Read input data (copies needed to avoid borrow conflicts)
            let lhs_data = ctx.tensor_registry.get_slice(*lhs)?.to_vec();
            let rhs_data = ctx.tensor_registry.get_slice(*rhs)?.to_vec();
            let len = lhs_data.len();

            // Allocate output
            ctx.tensor_registry.ensure_tensor(*output, len);
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;

            // Execute element-wise add using std::thread::scope
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::thread;

            let n_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            let counter = AtomicUsize::new(0);
            let chunk_size = 256;
            let out_addr = out_tensor.as_mut_slice().as_mut_ptr() as usize;

            thread::scope(|s| {
                for _ in 0..n_threads {
                    s.spawn(|| {
                        let out_ptr = out_addr as *mut f32;
                        loop {
                            let chunk_start = counter
                                .fetch_add(chunk_size, Ordering::Relaxed);
                            if chunk_start >= len {
                                break;
                            }
                            let chunk_end = (chunk_start + chunk_size).min(len);
                            for i in chunk_start..chunk_end {
                                unsafe {
                                    *out_ptr.add(i) = lhs_data[i] + rhs_data[i];
                                }
                            }
                        }
                    });
                }
            });

            Some(())
        }

        Operation::FusedQKVWeight {
            input,
            qkv_weight,
            output_q,
            output_k,
            output_v,
            q_out_size,
            k_out_size,
            v_out_size,
            layer_idx: _,
        } => {
            // Read input data (copy needed to avoid borrow conflicts with outputs)
            let input_data = ctx.tensor_registry.get_slice(*input)?.to_vec();
            let weight_tensor = ctx.weight_bindings.get(qkv_weight)?;

            // Allocate outputs
            ctx.tensor_registry.ensure_tensor(*output_q, *q_out_size);
            ctx.tensor_registry.ensure_tensor(*output_k, *k_out_size);
            ctx.tensor_registry.ensure_tensor(*output_v, *v_out_size);

            // Execute fused QKV projection
            // For CPU, we do a combined GEMV and split the result
            let total_out_size = q_out_size + k_out_size + v_out_size;
            let mut combined = vec![0.0f32; total_out_size];
            ctx.model.gemv_into(&input_data, weight_tensor, &mut combined);

            // Get output tensors and copy results
            let q_tensor = ctx.tensor_registry.get_mut(*output_q)?;
            q_tensor.as_mut_slice().copy_from_slice(&combined[..*q_out_size]);

            let k_tensor = ctx.tensor_registry.get_mut(*output_k)?;
            k_tensor.as_mut_slice().copy_from_slice(
                &combined[*q_out_size..*q_out_size + *k_out_size],
            );

            let v_tensor = ctx.tensor_registry.get_mut(*output_v)?;
            v_tensor
                .as_mut_slice()
                .copy_from_slice(&combined[*q_out_size + *k_out_size..]);

            Some(())
        }

        // Batch operations for prefill
        Operation::NoOp => {
            // Skip this node
            Some(())
        }

        _ => {
            // Unsupported operation
            eprintln!(
                "Warning: Unsupported operation in CPU DAG executor: {:?}",
                std::mem::discriminant(op)
            );
            eprintln!("  Operation name: {}", op_name);
            None
        }
    }
}

/// Extract output tensor ID from the last node in execution order
fn find_output_tensor_id(
    dag: &ComputeDag,
    execution_order: &[usize],
) -> Option<usize> {
    // Find the last node that produces logits (output projection)
    for &node_id in execution_order.iter().rev() {
        let node = &dag.nodes[node_id];
        match &node.op {
            Operation::Gemv { output, .. } => {
                // The last GEMV is the output projection
                return Some(*output);
            }
            _ => continue,
        }
    }
    None
}

/// CPU DAG Executor
#[allow(dead_code)]
pub struct CpuDagExecutor {
    tensor_registry: CpuTensorRegistry,
}

impl CpuDagExecutor {
    pub fn new() -> Self {
        Self { tensor_registry: CpuTensorRegistry::new() }
    }

    /// Execute a forward pass using the DAG
    pub fn forward(
        &mut self,
        model: &LlamaModel,
        tokens: &[u32],
        kv_cache: &mut KVCache,
        optimized_dag: &OptimizedDag,
        weight_bindings: &HashMap<usize, crate::model::Tensor>,
    ) -> Option<Vec<f32>> {
        // Create execution context
        let mut ctx = CpuExecutionContext {
            model,
            kv_cache,
            tensor_registry: CpuTensorRegistry::with_capacity(256),
            weight_bindings,
            layer_idx: 0,
            position: tokens.len().saturating_sub(1),
            k_tensor_ids: HashMap::new(),
            v_tensor_ids: HashMap::new(),
            buffers: None,
        };

        // Execute DAG nodes in execution order
        for node_id in &optimized_dag.execution_order {
            let node = &optimized_dag.dag.nodes[*node_id];
            execute_operation(&mut ctx, &node.op)?;
        }

        // Find output tensor ID from the last GEMV operation
        let output_id = find_output_tensor_id(
            &optimized_dag.dag,
            &optimized_dag.execution_order,
        )?;
        let output_tensor = ctx.tensor_registry.get_slice(output_id)?;

        // Return result
        Some(output_tensor.to_vec())
    }
}
