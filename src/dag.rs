// SPDX-License-Identifier: MIT

//! Compute graph (DAG) for inference

pub mod builder;

use std::collections::HashMap;

use crate::ops::{CoreOperator, QuantType};

/// Tensor identifier
pub type TensorId = usize;

/// Tensor in the compute graph
#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub name: String,
    pub dims: Vec<u64>,
    pub quant_type: QuantType,
    pub data: Option<Vec<u8>>,
    pub op: Option<OpType>,
    pub src: Vec<TensorId>,
}

/// Operation type that produces a tensor
#[derive(Debug, Clone)]
pub enum OpType {
    Input,
    Weight,
    Operator(CoreOperator),
}

/// Compute graph (DAG)
#[derive(Debug, Default)]
pub struct ComputeGraph {
    pub tensors: Vec<Tensor>,
    pub tensor_map: HashMap<String, TensorId>,
    pub execution_order: Vec<TensorId>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor and return its ID
    pub fn add_tensor(
        &mut self,
        name: &str,
        dims: Vec<u64>,
        quant_type: QuantType,
    ) -> TensorId {
        let id = self.tensors.len();
        let tensor = Tensor {
            id,
            name: name.to_string(),
            dims,
            quant_type,
            data: None,
            op: None,
            src: Vec::new(),
        };
        self.tensors.push(tensor);
        self.tensor_map.insert(name.to_string(), id);
        id
    }

    /// Add a weight tensor with data
    pub fn add_weight(
        &mut self,
        name: &str,
        dims: Vec<u64>,
        quant_type: QuantType,
        data: Vec<u8>,
    ) -> TensorId {
        let id = self.add_tensor(name, dims, quant_type);
        self.tensors[id].data = Some(data);
        self.tensors[id].op = Some(OpType::Weight);
        id
    }

    /// Add an input tensor
    pub fn add_input(
        &mut self,
        name: &str,
        dims: Vec<u64>,
        quant_type: QuantType,
    ) -> TensorId {
        let id = self.add_tensor(name, dims, quant_type);
        self.tensors[id].op = Some(OpType::Input);
        id
    }

    /// Add an operation that produces a tensor
    pub fn add_op(
        &mut self,
        name: &str,
        op: CoreOperator,
        src: Vec<TensorId>,
        dims: Vec<u64>,
        quant_type: QuantType,
    ) -> TensorId {
        let id = self.add_tensor(name, dims, quant_type);
        self.tensors[id].op = Some(OpType::Operator(op));
        self.tensors[id].src = src;
        id
    }

    /// Get tensor by ID
    pub fn get_tensor(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(id)
    }

    /// Get tensor by name
    pub fn get_tensor_by_name(&self, name: &str) -> Option<&Tensor> {
        self.tensor_map.get(name).and_then(|&id| self.tensors.get(id))
    }

    /// Build execution order (topological sort)
    pub fn build_execution_order(&mut self) -> Result<(), String> {
        let n = self.tensors.len();
        let mut visited = vec![false; n];
        let mut in_stack = vec![false; n];
        let mut order = Vec::with_capacity(n);

        for i in 0..n {
            if !visited[i] {
                self.topo_sort(i, &mut visited, &mut in_stack, &mut order)?;
            }
        }

        order.reverse();
        self.execution_order = order;
        Ok(())
    }

    fn topo_sort(
        &self,
        node: TensorId,
        visited: &mut [bool],
        in_stack: &mut [bool],
        order: &mut Vec<TensorId>,
    ) -> Result<(), String> {
        visited[node] = true;
        in_stack[node] = true;

        for &src in &self.tensors[node].src {
            if in_stack[src] {
                return Err(format!(
                    "Cycle detected: {}",
                    self.tensors[node].name
                ));
            }
            if !visited[src] {
                self.topo_sort(src, visited, in_stack, order)?;
            }
        }

        in_stack[node] = false;
        order.push(node);
        Ok(())
    }

    /// Get output tensors (not used as input to others)
    pub fn get_outputs(&self) -> Vec<TensorId> {
        let mut is_output = vec![true; self.tensors.len()];
        for tensor in &self.tensors {
            for &src in &tensor.src {
                is_output[src] = false;
            }
        }
        (0..self.tensors.len())
            .filter(|&i| is_output[i] && self.tensors[i].op.is_some())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_tensors() {
        let mut graph = ComputeGraph::new();

        let input = graph.add_input("input", vec![1, 4096], QuantType::F32);
        let weight = graph.add_weight(
            "weight",
            vec![4096, 4096],
            QuantType::F32,
            vec![0.0f32; 4096 * 4096]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        );

        assert_eq!(input, 0);
        assert_eq!(weight, 1);
        assert_eq!(graph.tensors.len(), 2);
    }

    #[test]
    fn test_execution_order() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_input("a", vec![10], QuantType::F32);
        let b = graph.add_input("b", vec![10], QuantType::F32);
        let c = graph.add_op(
            "c",
            CoreOperator::RmsNorm { hidden_size: 10, eps: 1e-5 },
            vec![a, b],
            vec![10],
            QuantType::F32,
        );

        graph.build_execution_order().unwrap();

        // a and b should come before c
        let a_pos =
            graph.execution_order.iter().position(|&id| id == a).unwrap();
        let b_pos =
            graph.execution_order.iter().position(|&id| id == b).unwrap();
        let c_pos =
            graph.execution_order.iter().position(|&id| id == c).unwrap();

        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_input("a", vec![10], QuantType::F32);
        let b = graph.add_op(
            "b",
            CoreOperator::RmsNorm { hidden_size: 10, eps: 1e-5 },
            vec![a],
            vec![10],
            QuantType::F32,
        );

        // Create a cycle: a -> b -> a
        graph.tensors[a].src.push(b);

        assert!(graph.build_execution_order().is_err());
    }
}
