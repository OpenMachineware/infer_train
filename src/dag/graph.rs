// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use crate::model::Tensor;
use std::collections::HashMap;
use std::rc::Rc;

/// Operation types in the DAG
#[derive(Debug, Clone)]
pub enum Operation {
    // Embedding
    EmbeddingLookup {
        token_id: u32,
        embedding_weight: usize,
        output: usize,
    },

    // Attention operations
    RmsNorm {
        input: usize,
        weight: usize,
        output: usize,
        epsilon: f32,
    },
    Gemv {
        input: usize,
        weight: usize,
        output: usize,
        out_size: usize,
    },
    /// Fused QKV projection: combines Q, K, V projections into single GEMM
    /// output_q, output_k, output_v are tensor IDs for each projection
    FusedQKV {
        input: usize,
        q_weight: usize,
        k_weight: usize,
        v_weight: usize,
        output_q: usize,
        output_k: usize,
        output_v: usize,
        q_out_size: usize,
        k_out_size: usize,
        v_out_size: usize,
    },
    RoPE {
        input: usize,
        output: usize,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
    },
    FlashAttention {
        q: usize,
        k_cache: usize,
        v_cache: usize,
        output: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_seq_len: usize,
    },
    KVCacheUpdate {
        input: usize,
        cache: usize,
        pos: usize,
    },

    // FFN operations
    SiLU {
        input: usize,
        output: usize,
    },
    Mul {
        lhs: usize,
        rhs: usize,
        output: usize,
    },
    Add {
        lhs: usize,
        rhs: usize,
        output: usize,
    },

    // Fused operations
    /// FusedSwiGLU: output = SiLU(gate) * up
    /// Combines SiLU activation and element-wise multiplication
    FusedSwiGLU {
        gate: usize, // Gate projection output
        up: usize,   // Up projection output
        output: usize,
    },
    /// FusedNormLinear: output = RMSNorm(input, norm_weight, eps) @ weights
    /// Combines RMSNorm and linear projection (GEMV)
    FusedNormLinear {
        input: usize,
        norm_weight: usize,   // RMSNorm scale weights
        linear_weight: usize, // Linear projection weights
        output: usize,
        out_size: usize, // M (output dimension)
        in_size: usize,  // K (input dimension)
        epsilon: f32,
    },

    /// Placeholder for removed/fused operations
    /// Executor skips these nodes
    NoOp,

    // MoE operations
    ExpertRouting {
        hidden: usize,
        router_weight: usize,
        selected_experts: usize,
        routing_weights: usize,
    },
    ExpertCompute {
        hidden: usize,
        expert_weights: Vec<usize>,
        routing_weights: usize,
        output: usize,
    },
}

/// DAG Node representing a single operation
#[derive(Debug, Clone)]
pub struct DagNode {
    pub id: usize,
    pub op: Operation,
    pub dependencies: Vec<usize>, // Node IDs this depends on
    pub device_preference: DevicePreference,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DevicePreference {
    Gpu, // Prefer GPU execution
    Cpu, // Prefer CPU execution
    Any, // No preference
}

/// DAG Edge representing data flow
#[derive(Debug, Clone)]
pub struct DagEdge {
    pub from_node: usize,
    pub to_node: usize,
    pub tensor_id: usize,
}

/// Complete DAG for a forward pass
#[derive(Debug)]
pub struct ComputeDag {
    pub nodes: Vec<DagNode>,
    pub edges: Vec<DagEdge>,
    pub tensor_registry: Vec<TensorLocation>,
    pub entry_points: Vec<usize>, // Node IDs with no dependencies
    pub exit_points: Vec<usize>,  // Node IDs that produce final output

    /// Weight tensor bindings: tensor_id -> actual Tensor
    /// Used at execution time to get GPU buffer for weights
    pub weight_bindings: HashMap<usize, Tensor>,
}

/// Where a tensor resides
#[derive(Debug, Clone)]
pub enum TensorLocation {
    Gpu(Rc<metal::Buffer>),
    Cpu(Box<[u8]>),
    NotYetAllocated,
}

impl Default for ComputeDag {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeDag {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            tensor_registry: Vec::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
            weight_bindings: HashMap::new(),
        }
    }

    /// Add a node to the DAG
    pub fn add_node(&mut self, node: DagNode) -> usize {
        let id = self.nodes.len();
        let has_no_deps = node.dependencies.is_empty();

        self.nodes.push(node);

        if has_no_deps {
            self.entry_points.push(id);
        }

        id
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: usize, to: usize, tensor_id: usize) {
        self.edges.push(DagEdge { from_node: from, to_node: to, tensor_id });
    }

    /// Bind a weight tensor to a tensor ID
    pub fn bind_weight(&mut self, tensor_id: usize, tensor: Tensor) {
        self.weight_bindings.insert(tensor_id, tensor);
    }

    /// Topological sort for execution order
    pub fn execution_order(&self) -> Vec<usize> {
        let mut in_degree = vec![0usize; self.nodes.len()];
        let mut result = Vec::with_capacity(self.nodes.len());

        // Build reverse dependency map: node -> nodes that depend on it
        let mut dependents: Vec<Vec<usize>> =
            vec![Vec::new(); self.nodes.len()];

        // Calculate in-degrees and build dependents map
        for node in &self.nodes {
            in_degree[node.id] = node.dependencies.len();
            for &dep in &node.dependencies {
                dependents[dep].push(node.id);
            }
        }

        // Start with nodes that have no dependencies
        let mut queue: Vec<usize> = self.entry_points.clone();

        while let Some(node_id) = queue.pop() {
            result.push(node_id);

            // Decrement in-degree for all dependents
            for &dependent in &dependents[node_id] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    queue.push(dependent);
                }
            }
        }

        result
    }

    /// Convert to OptimizedDag with pre-computed execution order
    pub fn into_optimized(self) -> OptimizedDag {
        let execution_order = self.execution_order();
        OptimizedDag { dag: self, execution_order }
    }
}

/// Optimized DAG with pre-computed execution order
///
/// This caches the topological sort result, avoiding O(V+E) computation
/// on every forward pass.
#[derive(Debug)]
pub struct OptimizedDag {
    pub dag: ComputeDag,
    pub execution_order: Vec<usize>,
}

impl OptimizedDag {
    /// Iterate over nodes in execution order
    pub fn nodes_in_order(&self) -> impl Iterator<Item = &DagNode> {
        self.execution_order.iter().map(move |&id| &self.dag.nodes[id])
    }

    /// Get total node count
    pub fn node_count(&self) -> usize {
        self.dag.nodes.len()
    }

    /// Get total edge count
    pub fn edge_count(&self) -> usize {
        self.dag.edges.len()
    }
}
