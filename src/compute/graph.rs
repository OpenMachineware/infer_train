// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compute graph for tensor operations

use std::collections::HashMap;

use super::op::{OpNode, OpParams, OpType};
use super::tensor::Tensor;

/// A compute graph containing tensors and operations
#[derive(Debug, Default)]
pub struct ComputeGraph {
    /// Named tensors (inputs, outputs, intermediates)
    pub tensors: HashMap<String, Tensor>,
    /// Operations in execution order
    pub nodes: Vec<OpNode>,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Scratch buffers (reusable memory)
    scratch_sizes: HashMap<String, usize>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor to the graph
    pub fn add_tensor(&mut self, tensor: Tensor) -> &mut Self {
        let name = tensor.name.clone();
        self.tensors.insert(name, tensor);
        self
    }

    /// Mark a tensor as an input
    pub fn add_input(&mut self, name: &str) -> &mut Self {
        if !self.inputs.contains(&name.to_string()) {
            self.inputs.push(name.to_string());
        }
        self
    }

    /// Mark a tensor as an output
    pub fn add_output(&mut self, name: &str) -> &mut Self {
        if !self.outputs.contains(&name.to_string()) {
            self.outputs.push(name.to_string());
        }
        self
    }

    /// Add an operation node
    pub fn add_op(&mut self, node: OpNode) -> &mut Self {
        self.nodes.push(node);
        self
    }

    /// Add an operation with automatic node creation
    pub fn op(
        &mut self,
        op_type: OpType,
        inputs: Vec<String>,
        output: String,
    ) -> &mut Self {
        self.add_op(OpNode::new(op_type, inputs, output))
    }

    /// Add an operation with parameters
    pub fn op_with_params(
        &mut self,
        op_type: OpType,
        inputs: Vec<String>,
        output: String,
        params: OpParams,
    ) -> &mut Self {
        self.add_op(OpNode::new(op_type, inputs, output).with_params(params))
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get a mutable tensor by name
    pub fn get_tensor_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.tensors.get_mut(name)
    }

    /// Check if all input tensors are available
    pub fn validate_inputs(&self) -> Result<(), String> {
        for name in &self.inputs {
            if !self.tensors.contains_key(name) {
                return Err(format!("Missing input tensor: {}", name));
            }
        }
        Ok(())
    }

    /// Get execution order (topological sort)
    /// For now, nodes are already in order. Future: auto-toposort.
    pub fn execution_order(&self) -> &[OpNode] {
        &self.nodes
    }

    /// Allocate scratch buffer
    pub fn alloc_scratch(&mut self, name: &str, size: usize) {
        self.scratch_sizes.insert(name.to_string(), size);
    }

    /// Get scratch buffer size
    pub fn scratch_size(&self, name: &str) -> Option<usize> {
        self.scratch_sizes.get(name).copied()
    }

    /// Clear the graph (keep allocated tensors)
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.inputs.clear();
        self.outputs.clear();
    }

    /// Get number of operations
    pub fn op_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get total tensor count
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

/// Builder for compute graphs with chainable API
pub struct GraphBuilder {
    graph: ComputeGraph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { graph: ComputeGraph::new() }
    }

    pub fn tensor(mut self, tensor: Tensor) -> Self {
        self.graph.add_tensor(tensor);
        self
    }

    pub fn input(mut self, name: &str) -> Self {
        self.graph.add_input(name);
        self
    }

    pub fn output(mut self, name: &str) -> Self {
        self.graph.add_output(name);
        self
    }

    pub fn op(
        mut self,
        op_type: OpType,
        inputs: Vec<String>,
        output: String,
    ) -> Self {
        self.graph.op(op_type, inputs, output);
        self
    }

    pub fn op_with_params(
        mut self,
        op_type: OpType,
        inputs: Vec<String>,
        output: String,
        params: OpParams,
    ) -> Self {
        self.graph.op_with_params(op_type, inputs, output, params);
        self
    }

    pub fn build(self) -> ComputeGraph {
        self.graph
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}
