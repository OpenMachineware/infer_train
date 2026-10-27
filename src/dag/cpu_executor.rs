// SPDX-License-Identifier: AGPL-3.0-or-later
//
// CPU DAG Executor - Execute DAG operations on CPU with zero allocation
//
// This executor reuses the existing DAG infrastructure for CPU path,
// providing graph plan optimization benefits to CPU inference.

use super::graph::{ComputeDag, DagNode, Operation, OptimizedDag};
use crate::inference::{KVCache, KVCacheBase};
use crate::model::LlamaModel;
use std::collections::HashMap;

/// CPU tensor: pre-allocated buffer with size tracking
#[derive(Clone)]
pub struct CpuTensor {
    pub data: Vec<f32>,
    pub len: usize, // Actual length (may be less than capacity)
}

impl CpuTensor {
    pub fn new(size: usize) -> Self {
        CpuTensor { data: vec![0.0f32; size], len: size }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data[..self.len]
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

    /// Create or resize a tensor
    pub fn ensure_tensor(&mut self, id: usize, size: usize) {
        if id >= self.tensors.len() {
            let new_len = (id + 1).next_power_of_two();
            self.tensors.resize(new_len, None);
        }

        match &mut self.tensors[id] {
            Some(tensor) => {
                if tensor.data.len() < size {
                    tensor.data.resize(size, 0.0);
                    tensor.len = size;
                }
            }
            None => {
                self.tensors[id] = Some(CpuTensor::new(size));
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
            // Read input data (copy needed to avoid borrow conflict with output)
            let input_data = ctx.tensor_registry.get_slice(*input)?.to_vec();
            let weight_tensor = ctx.weight_bindings.get(weight)?;

            if *is_transposed {
                // Transposed GEMV: weight is stored as [in_size, out_size]
                let result = ctx.model.gemv_transposed(&input_data, weight_tensor, *out_size)?;
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

        Operation::RoPE { input, output, pos, n_heads, head_dim } => {
            // Read input data (copy needed to avoid borrow conflict with output)
            let input_data = ctx.tensor_registry.get_slice(*input)?.to_vec();

            // Allocate output
            ctx.tensor_registry.ensure_tensor(*output, input_data.len());
            let out_tensor = ctx.tensor_registry.get_mut(*output)?;

            // Execute RoPE
            ctx.model.apply_rope_into(
                &input_data,
                *pos,
                *n_heads,
                *head_dim,
                out_tensor.as_mut_slice(),
            )?;

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
