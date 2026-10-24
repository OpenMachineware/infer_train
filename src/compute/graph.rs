// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compute graph for unified execution across backends.

use std::collections::HashMap;

use super::storage::Storage;
use crate::model::tensor::DataType;

/// Operation type in the compute graph.
#[derive(Debug, Clone)]
pub enum OpType {
    // Normalization
    RmsNorm {
        eps: f32,
    },
    LayerNorm {
        eps: f32,
    },

    // Matrix multiplication
    MatMul,
    MatMulId, // MoE expert routing

    // Attention
    FlashAttn {
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
    },
    Softmax,

    // Position encoding
    RopeNeox {
        n_heads: usize,
        head_dim: usize,
        rope_freq_base: f32,
        rope_freq_scale: f32,
    },

    // Activation
    Silu,
    Gelu,
    Swiglu, // w1 * silu(w3) for FFN

    // Element-wise
    Add,
    Mul,
    Scale {
        factor: f32,
    },

    // KV cache
    KVStore,
    KVLoad,

    // Embedding
    EmbeddingLookup,

    // Reduction
    Sum,
    Mean,
}

/// A node in the compute graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Operation type.
    pub op: OpType,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor name.
    pub output: String,
}

/// Tensor in the compute graph.
#[derive(Debug, Clone)]
pub struct GraphTensor {
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DataType,
    /// Storage (None for intermediate tensors to be allocated).
    pub storage: Option<Storage>,
}

/// Compute graph for a forward pass.
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    /// Named tensors (weights, inputs, intermediates).
    pub tensors: HashMap<String, GraphTensor>,
    /// Operations in execution order.
    pub nodes: Vec<OpNode>,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
}

impl ComputeGraph {
    /// Create an empty compute graph.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a tensor to the graph.
    pub fn add_tensor(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        dtype: DataType,
        storage: Option<Storage>,
    ) {
        self.tensors
            .insert(name.to_string(), GraphTensor { shape, dtype, storage });
    }

    /// Mark a tensor as an input.
    pub fn add_input(&mut self, name: &str) {
        if !self.inputs.contains(&name.to_string()) {
            self.inputs.push(name.to_string());
        }
    }

    /// Mark a tensor as an output.
    pub fn add_output(&mut self, name: &str) {
        if !self.outputs.contains(&name.to_string()) {
            self.outputs.push(name.to_string());
        }
    }

    /// Add an operation node.
    pub fn add_op(&mut self, op: OpType, inputs: Vec<String>, output: String) {
        // Ensure output tensor exists (will be allocated during execution)
        if !self.tensors.contains_key(&output) {
            self.tensors.insert(
                output.clone(),
                GraphTensor {
                    shape: vec![],        // Will be inferred during execution
                    dtype: DataType::F32, // Default, can be overridden
                    storage: None,
                },
            );
        }

        self.nodes.push(OpNode { op, inputs, output });
    }

    /// Get tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GraphTensor> {
        self.tensors.get(name)
    }

    /// Get mutable tensor by name.
    pub fn get_tensor_mut(&mut self, name: &str) -> Option<&mut GraphTensor> {
        self.tensors.get_mut(name)
    }

    /// Check if graph is valid (all inputs available, no cycles).
    pub fn validate(&self) -> Result<(), String> {
        // Check all inputs exist
        for input in &self.inputs {
            if !self.tensors.contains_key(input) {
                return Err(format!("Input tensor '{}' not found", input));
            }
        }

        // Check all operation inputs are available
        let mut available: std::collections::HashSet<String> =
            self.inputs.iter().cloned().collect();

        for (i, node) in self.nodes.iter().enumerate() {
            for input in &node.inputs {
                if !available.contains(input)
                    && !self.tensors.contains_key(input)
                {
                    return Err(format!(
                        "Node {}: input '{}' not available",
                        i, input
                    ));
                }
            }
            available.insert(node.output.clone());
        }

        // Check outputs are produced
        for output in &self.outputs {
            if !available.contains(output) {
                return Err(format!(
                    "Output '{}' not produced by any node",
                    output
                ));
            }
        }

        Ok(())
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating compute graphs.
pub struct GraphBuilder {
    graph: ComputeGraph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { graph: ComputeGraph::new() }
    }

    /// Add a weight tensor with storage.
    pub fn weight(
        mut self,
        name: &str,
        shape: Vec<usize>,
        dtype: DataType,
        storage: Storage,
    ) -> Self {
        self.graph.add_tensor(name, shape, dtype, Some(storage));
        self
    }

    /// Add an input tensor (no storage, will be provided at runtime).
    pub fn input(
        mut self,
        name: &str,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> Self {
        self.graph.add_tensor(name, shape, dtype, None);
        self.graph.add_input(name);
        self
    }

    /// Add RMS normalization.
    pub fn rms_norm(
        mut self,
        input: &str,
        weight: &str,
        output: &str,
        eps: f32,
    ) -> Self {
        self.graph.add_op(
            OpType::RmsNorm { eps },
            vec![input.to_string(), weight.to_string()],
            output.to_string(),
        );
        self
    }

    /// Add matrix multiplication.
    pub fn matmul(mut self, a: &str, b: &str, output: &str) -> Self {
        self.graph.add_op(
            OpType::MatMul,
            vec![a.to_string(), b.to_string()],
            output.to_string(),
        );
        self
    }

    /// Add RoPE position encoding.
    pub fn rope(
        mut self,
        input: &str,
        _pos: usize,
        output: &str,
        n_heads: usize,
        head_dim: usize,
        freq_base: f32,
    ) -> Self {
        // Position is passed via a constant tensor
        let pos_name = format!("{}_pos", input);
        self.graph.add_tensor(&pos_name, vec![1], DataType::F32, None);

        self.graph.add_op(
            OpType::RopeNeox {
                n_heads,
                head_dim,
                rope_freq_base: freq_base,
                rope_freq_scale: 1.0,
            },
            vec![input.to_string(), pos_name.clone()],
            output.to_string(),
        );
        self
    }

    /// Add Flash Attention.
    pub fn flash_attn(
        mut self,
        q: &str,
        k_cache: &str,
        v_cache: &str,
        output: &str,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        max_seq: usize,
    ) -> Self {
        self.graph.add_op(
            OpType::FlashAttn {
                n_heads,
                head_dim,
                n_kv_heads,
                max_seq_len: max_seq,
            },
            vec![q.to_string(), k_cache.to_string(), v_cache.to_string()],
            output.to_string(),
        );
        self
    }

    /// Add SwiGLU FFN.
    pub fn swiglu(
        mut self,
        input: &str,
        w1: &str,
        w2: &str,
        w3: &str,
        output: &str,
    ) -> Self {
        // w1 * silu(w3) -> intermediate
        let gate = format!("{}_gate", input);
        let up = format!("{}_up", input);

        // w1 @ input -> up
        self.graph.add_op(
            OpType::MatMul,
            vec![input.to_string(), w1.to_string()],
            up.clone(),
        );

        // w3 @ input -> gate
        self.graph.add_op(
            OpType::MatMul,
            vec![input.to_string(), w3.to_string()],
            gate.clone(),
        );

        // silu(gate)
        let gate_silu = format!("{}_silu", gate);
        self.graph.add_op(OpType::Silu, vec![gate], gate_silu.clone());

        // up * gate_silu
        let ffn_intermediate = format!("{}_ffn_mid", input);
        self.graph.add_op(
            OpType::Mul,
            vec![up, gate_silu],
            ffn_intermediate.clone(),
        );

        // w2 @ intermediate -> output
        self.graph.add_op(
            OpType::MatMul,
            vec![ffn_intermediate, w2.to_string()],
            output.to_string(),
        );

        self
    }

    /// Add residual connection.
    pub fn add(mut self, a: &str, b: &str, output: &str) -> Self {
        self.graph.add_op(
            OpType::Add,
            vec![a.to_string(), b.to_string()],
            output.to_string(),
        );
        self
    }

    /// Mark output tensor.
    pub fn output(mut self, name: &str) -> Self {
        self.graph.add_output(name);
        self
    }

    /// Build the compute graph.
    pub fn build(self) -> Result<ComputeGraph, String> {
        self.graph.validate()?;
        Ok(self.graph)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}
