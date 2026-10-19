// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! DAG Optimizer - Platform-independent fusion
//!
//! Applies optimizations to the DAG before execution

use super::graph::{ComputeDag, DagNode, DevicePreference, Operation};

pub struct DagOptimizer;

impl DagOptimizer {
    pub fn new() -> Self {
        Self
    }

    /// Find and apply all fusion patterns to the DAG
    /// Returns the number of nodes that were fused
    pub fn optimize(dag: &mut ComputeDag) -> usize {
        let mut fused_count = 0;

        // Apply SiLU + Mul → FusedSwiGLU fusion
        fused_count += Self::fuse_silu_mul(dag);

        // Apply RMSNorm + Gemv → FusedNormLinear fusion
        fused_count += Self::fuse_norm_gemv(dag);

        fused_count
    }

    /// Fuse SiLU + Mul → FusedSwiGLU
    fn fuse_silu_mul(dag: &mut ComputeDag) -> usize {
        let mut fused_count = 0;
        let mut nodes_to_remove = Vec::new();
        let mut nodes_to_add = Vec::new();

        for silu_node in &dag.nodes {
            if let Operation::SiLU { input: silu_input, output: silu_output } =
                &silu_node.op
            {
                // Find Mul that uses this SiLU output
                for mul_node in &dag.nodes {
                    if mul_node.id <= silu_node.id {
                        continue; // Process in order
                    }
                    if let Operation::Mul { lhs, rhs, output: mul_output } =
                        &mul_node.op
                    {
                        if *lhs == *silu_output || *rhs == *silu_output {
                            // Found SiLU + Mul pattern
                            let up_input =
                                if *lhs == *silu_output { *rhs } else { *lhs };

                            // Mark these nodes for removal
                            nodes_to_remove.push(silu_node.id);
                            nodes_to_remove.push(mul_node.id);

                            // Create fused node with combined dependencies
                            // Exclude the nodes being removed from dependencies
                            let mut combined_deps = Vec::new();
                            for &dep in &silu_node.dependencies {
                                if !nodes_to_remove.contains(&dep) {
                                    combined_deps.push(dep);
                                }
                            }
                            for &dep in &mul_node.dependencies {
                                if !combined_deps.contains(&dep)
                                    && !nodes_to_remove.contains(&dep)
                                {
                                    combined_deps.push(dep);
                                }
                            }

                            let fused_node = DagNode {
                                id: dag.nodes.len() + nodes_to_add.len(), // Temporary ID
                                op: Operation::FusedSwiGLU {
                                    gate: *silu_input,
                                    up: up_input,
                                    output: *mul_output,
                                },
                                dependencies: combined_deps,
                                device_preference: DevicePreference::Gpu,
                            };

                            nodes_to_add.push(fused_node);
                            fused_count += 2;
                            break;
                        }
                    }
                }
            }
        }

        // Apply modifications
        Self::apply_modifications(dag, nodes_to_remove, nodes_to_add);

        fused_count
    }

    /// Fuse RMSNorm + Gemv → FusedNormLinear
    /// Only fuses when the norm output has exactly ONE consumer (the Gemv)
    fn fuse_norm_gemv(dag: &mut ComputeDag) -> usize {
        let mut fused_count = 0;
        let mut nodes_to_remove = Vec::new();
        let mut nodes_to_add = Vec::new();

        for norm_node in &dag.nodes {
            if let Operation::RmsNorm {
                input: norm_input,
                output: norm_output,
                weight: norm_weight,
                epsilon,
            } = &norm_node.op
            {
                // Count how many Gemv nodes use this norm output
                let consumers: Vec<_> = dag
                    .nodes
                    .iter()
                    .filter_map(|node| {
                        if let Operation::Gemv { input, .. } = &node.op {
                            if *input == *norm_output {
                                Some(node.id)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                // Only fuse if there's exactly ONE consumer
                if consumers.len() != 1 {
                    continue;
                }

                let gemv_node_id = consumers[0];
                let gemv_node = &dag.nodes[gemv_node_id];

                if let Operation::Gemv {
                    input: gemv_input,
                    weight: gemv_weight,
                    output: gemv_output,
                    out_size,
                } = &gemv_node.op
                {
                    // Verify the input matches
                    if *gemv_input != *norm_output {
                        continue;
                    }

                    // Mark these nodes for removal
                    nodes_to_remove.push(norm_node.id);
                    nodes_to_remove.push(gemv_node_id);

                    // Create fused node with combined dependencies
                    // Exclude the nodes being removed from dependencies
                    let mut combined_deps = Vec::new();
                    for &dep in &norm_node.dependencies {
                        if !nodes_to_remove.contains(&dep) {
                            combined_deps.push(dep);
                        }
                    }
                    for &dep in &gemv_node.dependencies {
                        if !combined_deps.contains(&dep)
                            && !nodes_to_remove.contains(&dep)
                        {
                            combined_deps.push(dep);
                        }
                    }

                    let fused_node = DagNode {
                        id: dag.nodes.len() + nodes_to_add.len(), // Temporary ID
                        op: Operation::FusedNormLinear {
                            input: *norm_input,
                            norm_weight: *norm_weight,
                            linear_weight: *gemv_weight,
                            output: *gemv_output,
                            out_size: *out_size,
                            in_size: 0, // Will be filled from tensor shape
                            epsilon: *epsilon,
                        },
                        dependencies: combined_deps,
                        device_preference: DevicePreference::Gpu,
                    };

                    nodes_to_add.push(fused_node);
                    fused_count += 2;
                }
            }
        }

        // Apply modifications
        Self::apply_modifications(dag, nodes_to_remove, nodes_to_add);

        fused_count
    }

    /// Apply node removal and addition to the DAG
    fn apply_modifications(
        dag: &mut ComputeDag,
        nodes_to_remove: Vec<usize>,
        nodes_to_add: Vec<DagNode>,
    ) {
        if nodes_to_remove.is_empty() && nodes_to_add.is_empty() {
            return;
        }

        // Mark nodes for removal by replacing with NoOp
        // Keep dependencies intact to preserve topological ordering
        for node_id in nodes_to_remove {
            if let Some(node) = dag.nodes.get_mut(node_id) {
                node.op = Operation::NoOp;
                // Don't clear dependencies - they're needed for topological sort
            }
        }

        // Add new fused nodes
        for mut node in nodes_to_add {
            // Assign proper ID
            node.id = dag.nodes.len();
            dag.nodes.push(node);
        }
    }
}
