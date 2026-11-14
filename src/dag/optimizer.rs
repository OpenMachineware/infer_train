// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// DAG optimizer for fusion and scheduling

use crate::dag::{ComputeGraph, Op, TensorId};
use crate::types::QuantType;

// Fusion pattern: identifies sequences of ops that can be fused
pub struct DagOptimizer {
    graph: ComputeGraph,
}

impl DagOptimizer {
    pub fn new(graph: ComputeGraph) -> Self {
        Self { graph }
    }

    // Analyze and apply all fusion patterns
    pub fn optimize(mut self) -> ComputeGraph {
        self.fuse_rmsnorm_linear();
        self.fuse_silu_mul();
        self.graph.build_execution_order();
        self.graph
    }

    // Pattern: RmsNorm + Linear (GEMV) -> FusedNormLinear
    fn fuse_rmsnorm_linear(&mut self) {
        let candidates: Vec<(TensorId, TensorId, Vec<u64>, QuantType)> =
            self.find_rmsnorm_linear_pairs();
        for (norm_id, linear_id, dims, quant_type) in candidates {
            self.apply_rmsnorm_linear_fusion(
                norm_id, linear_id, &dims, quant_type,
            );
        }
    }

    // Pattern: Silu + Mul -> FusedSwiGLU
    fn fuse_silu_mul(&mut self) {
        let candidates: Vec<(
            TensorId,
            TensorId,
            Vec<u64>,
            QuantType,
            TensorId,
            TensorId,
        )> = self.find_silu_mul_pairs();
        for (silu_id, mul_id, dims, quant_type, silu_input, other_input) in
            candidates
        {
            self.apply_silu_mul_fusion(
                silu_id,
                mul_id,
                &dims,
                quant_type,
                silu_input,
                other_input,
            );
        }
    }

    fn find_rmsnorm_linear_pairs(
        &self,
    ) -> Vec<(TensorId, TensorId, Vec<u64>, QuantType)> {
        let mut pairs = Vec::new();
        for norm_id in 0..self.graph.tensors.len() {
            let norm = &self.graph.tensors[norm_id];
            if norm.op != Op::RmsNorm {
                continue;
            }
            // Find consumers that are MulMat
            for consumer_id in 0..self.graph.tensors.len() {
                let consumer = &self.graph.tensors[consumer_id];
                if consumer.op != Op::MulMat {
                    continue;
                }
                if consumer.src[0] == Some(norm_id) {
                    let dims: Vec<u64> =
                        consumer.dims[..consumer.n_dims].to_vec();
                    pairs.push((
                        norm_id,
                        consumer_id,
                        dims,
                        consumer.quant_type,
                    ));
                }
            }
        }
        pairs
    }

    fn find_silu_mul_pairs(
        &self,
    ) -> Vec<(TensorId, TensorId, Vec<u64>, QuantType, TensorId, TensorId)>
    {
        let mut pairs = Vec::new();
        for silu_id in 0..self.graph.tensors.len() {
            let silu = &self.graph.tensors[silu_id];
            if silu.op != Op::Silu {
                continue;
            }
            let silu_input = match silu.src[0] {
                Some(id) => id,
                None => continue,
            };

            // Find Mul that uses Silu output
            for mul_id in 0..self.graph.tensors.len() {
                let mul = &self.graph.tensors[mul_id];
                if mul.op != Op::Mul {
                    continue;
                }
                let other_input = if mul.src[0] == Some(silu_id) {
                    mul.src[1]
                } else if mul.src[1] == Some(silu_id) {
                    mul.src[0]
                } else {
                    continue;
                };

                if let Some(other) = other_input {
                    let dims: Vec<u64> = mul.dims[..mul.n_dims].to_vec();
                    pairs.push((
                        silu_id,
                        mul_id,
                        dims,
                        mul.quant_type,
                        silu_input,
                        other,
                    ));
                }
            }
        }
        pairs
    }

    fn apply_rmsnorm_linear_fusion(
        &mut self,
        norm_id: TensorId,
        linear_id: TensorId,
        dims: &[u64],
        quant_type: QuantType,
    ) {
        let linear = &self.graph.tensors[linear_id];
        let fused_name = format!("fused_norm_linear_{}", linear.name.clone());

        // Get norm input and weight
        let norm = &self.graph.tensors[norm_id];
        let input = match norm.src[0] {
            Some(id) => id,
            None => return,
        };
        let weight = match linear.src[1] {
            Some(id) => id,
            None => return,
        };

        // Create fused op
        let fused_id = self.graph.add_op(
            &fused_name,
            Op::FusedNormLinear,
            &[input, weight],
            dims,
            quant_type,
        );

        // Mark original ops as removed
        self.graph.tensors[norm_id].op = Op::None;
        self.graph.tensors[linear_id].op = Op::None;

        // Update outputs
        if let Some(pos) =
            self.graph.outputs.iter().position(|&id| id == linear_id)
        {
            self.graph.outputs[pos] = fused_id;
        }
    }

    fn apply_silu_mul_fusion(
        &mut self,
        silu_id: TensorId,
        mul_id: TensorId,
        dims: &[u64],
        quant_type: QuantType,
        silu_input: TensorId,
        other_input: TensorId,
    ) {
        let mul = &self.graph.tensors[mul_id];
        let fused_name = format!("fused_swiglu_{}", mul.name.clone());

        // Create fused op
        let fused_id = self.graph.add_op(
            &fused_name,
            Op::FusedSwiGLU,
            &[silu_input, other_input],
            dims,
            quant_type,
        );

        // Mark original ops as removed
        self.graph.tensors[silu_id].op = Op::None;
        self.graph.tensors[mul_id].op = Op::None;

        // Update outputs
        if let Some(pos) =
            self.graph.outputs.iter().position(|&id| id == mul_id)
        {
            self.graph.outputs[pos] = fused_id;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuantType;

    #[test]
    fn test_fusion_detection() {
        let mut graph = ComputeGraph::new();

        let input = graph.add_tensor("input", &[4096], QuantType::F32);
        let norm = graph.add_op(
            "norm",
            Op::RmsNorm,
            &[input],
            &[4096],
            QuantType::F32,
        );
        let weight = graph.add_tensor("weight", &[4096, 4096], QuantType::F32);
        let linear = graph.add_op(
            "linear",
            Op::MulMat,
            &[norm, weight],
            &[4096],
            QuantType::F32,
        );

        graph.outputs.push(linear);

        let optimizer = DagOptimizer::new(graph);
        let optimized = optimizer.optimize();

        // Check that fusion happened
        let has_fused =
            optimized.tensors.iter().any(|t| t.op == Op::FusedNormLinear);
        assert!(has_fused);
    }
}
