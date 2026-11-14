// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compute graph (DAG) for inference

pub mod builder;
pub mod optimizer;

use std::collections::HashMap;

use crate::types::QuantType;

// Tensor identifier
pub type TensorId = usize;

// Maximum number of source tensors
pub const MAX_SRCS: usize = 10;

// Operation type (like ggml_op)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    None,
    // Arithmetic
    Add,
    Mul,
    Div,
    // Matrix
    MulMat,
    MulMatId,
    // Normalization
    RmsNorm,
    LayerNorm,
    // Activation
    Silu,
    Gelu,
    SoftMax,
    // Attention
    RoPE,
    FlashAttn,
    // Embedding
    Embedding,
    // KV Cache
    KVUpdate,
    // Fused operations
    FusedAddNorm,
    FusedNormLinear,
    FusedSwiGLU,
}

// Tensor in the compute graph (like ggml_tensor)
#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub name: String,
    pub dims: [u64; 4],
    pub n_dims: usize,
    pub quant_type: QuantType,
    pub op: Op,
    pub src: [Option<TensorId>; MAX_SRCS],
    pub n_src: usize,
    // Data pointer (not owned, points to GGUF memory)
    pub data_offset: Option<u64>,
    pub data_size: usize,
}

impl Tensor {
    pub fn new(
        id: TensorId,
        name: String,
        dims: &[u64],
        quant_type: QuantType,
    ) -> Self {
        let mut d = [1u64; 4];
        for (i, &dim) in dims.iter().enumerate() {
            d[i] = dim;
        }
        Self {
            id,
            name,
            dims: d,
            n_dims: dims.len(),
            quant_type,
            op: Op::None,
            src: [None; MAX_SRCS],
            n_src: 0,
            data_offset: None,
            data_size: 0,
        }
    }

    pub fn add_src(&mut self, src: TensorId) {
        if self.n_src < MAX_SRCS {
            self.src[self.n_src] = Some(src);
            self.n_src += 1;
        }
    }

    pub fn nelements(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        let bytes_per_elem = match self.quant_type {
            QuantType::F32 => 4,
            QuantType::F16 | QuantType::BF16 => 2,
            _ => 1, // quantized, block-based
        };
        (self.nelements() * bytes_per_elem) as usize
    }
}

// Compute graph (like ggml_cgraph)
#[derive(Debug, Default)]
pub struct ComputeGraph {
    pub tensors: Vec<Tensor>,
    pub tensor_map: HashMap<String, TensorId>,
    pub execution_order: Vec<TensorId>,
    // Output tensors
    pub outputs: Vec<TensorId>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(
        &mut self,
        name: &str,
        dims: &[u64],
        quant_type: QuantType,
    ) -> TensorId {
        let id = self.tensors.len();
        let tensor = Tensor::new(id, name.to_string(), dims, quant_type);
        self.tensors.push(tensor);
        self.tensor_map.insert(name.to_string(), id);
        id
    }

    pub fn add_weight(
        &mut self,
        name: &str,
        dims: &[u64],
        quant_type: QuantType,
        offset: u64,
    ) -> TensorId {
        let id = self.add_tensor(name, dims, quant_type);
        self.tensors[id].data_offset = Some(offset);
        self.tensors[id].data_size = self.tensors[id].nbytes();
        id
    }

    pub fn add_op(
        &mut self,
        name: &str,
        op: Op,
        srcs: &[TensorId],
        dims: &[u64],
        quant_type: QuantType,
    ) -> TensorId {
        let id = self.add_tensor(name, dims, quant_type);
        self.tensors[id].op = op;
        for &src in srcs {
            self.tensors[id].add_src(src);
        }
        id
    }

    pub fn get_tensor_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        self.tensors.get_mut(id)
    }

    pub fn get_tensor(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(id)
    }

    pub fn get_tensor_by_name(&self, name: &str) -> Option<&Tensor> {
        self.tensor_map.get(name).and_then(|&id| self.tensors.get(id))
    }

    pub fn build_execution_order(&mut self) {
        let n = self.tensors.len();
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);

        for &output in &self.outputs {
            self.topo_sort(output, &mut visited, &mut order);
        }

        order.reverse();
        self.execution_order = order;
    }

    fn topo_sort(
        &self,
        node: TensorId,
        visited: &mut [bool],
        order: &mut Vec<TensorId>,
    ) {
        if visited[node] {
            return;
        }
        visited[node] = true;

        let tensor = &self.tensors[node];
        for i in 0..tensor.n_src {
            if let Some(src) = tensor.src[i] {
                self.topo_sort(src, visited, order);
            }
        }

        order.push(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dims() {
        let t = Tensor::new(0, "test".to_string(), &[2, 3, 4], QuantType::F32);
        assert_eq!(t.nelements(), 24);
        assert_eq!(t.nbytes(), 96);
    }

    #[test]
    fn test_graph_build() {
        let mut graph = ComputeGraph::new();

        let a = graph.add_tensor("a", &[10], QuantType::F32);
        let b = graph.add_tensor("b", &[10], QuantType::F32);
        let c = graph.add_op("c", Op::Add, &[a, b], &[10], QuantType::F32);

        graph.outputs.push(c);
        graph.build_execution_order();

        assert_eq!(graph.execution_order.len(), 3);
        assert!(graph.execution_order.contains(&a));
        assert!(graph.execution_order.contains(&b));
        assert!(graph.execution_order.contains(&c));
    }
}
