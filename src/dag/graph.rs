// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use crate::model::Tensor;
use std::rc::Rc;

/// Operation types in the DAG
#[derive(Debug, Clone)]
pub enum Operation {
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
}

/// Where a tensor resides
#[derive(Debug, Clone)]
pub enum TensorLocation {
    Gpu(Rc<metal::Buffer>),
    Cpu(Box<[u8]>),
    NotYetAllocated,
}

impl ComputeDag {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            tensor_registry: Vec::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
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

    /// Topological sort for execution order
    pub fn execution_order(&self) -> Vec<usize> {
        let mut in_degree = vec![0usize; self.nodes.len()];
        let mut result = Vec::with_capacity(self.nodes.len());

        // Calculate in-degrees
        for node in &self.nodes {
            in_degree[node.id] = node.dependencies.len();
        }

        // Start with nodes that have no dependencies
        let mut queue: Vec<usize> = self.entry_points.clone();

        while let Some(node_id) = queue.pop() {
            result.push(node_id);

            // Find all edges from this node
            for edge in &self.edges {
                if edge.from_node == node_id {
                    let to_node = edge.to_node;
                    in_degree[to_node] -= 1;

                    if in_degree[to_node] == 0 {
                        queue.push(to_node);
                    }
                }
            }
        }

        result
    }
}
