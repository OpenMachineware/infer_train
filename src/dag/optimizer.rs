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

        // Apply QKV projection fusion (most impactful)
        fused_count += Self::fuse_qkv_projections(dag);

        // Apply gate-up projection fusion
        fused_count += Self::fuse_gate_up_projections(dag);

        // Apply SiLU + Mul → FusedSwiGLU fusion
        fused_count += Self::fuse_silu_mul(dag);

        // Apply RMSNorm + Gemv → FusedNormLinear fusion
        fused_count += Self::fuse_norm_gemv(dag);

        // Apply Add + RMSNorm → FusedAddNorm fusion (residual + FFN norm)
        fused_count += Self::fuse_add_norm(dag);

        fused_count
    }

    /// Fuse Q, K, V projections into FusedQKV
    /// Finds three GEMV operations that all use the same input tensor
    fn fuse_qkv_projections(dag: &mut ComputeDag) -> usize {
        let mut fused_count = 0;
        let mut nodes_to_remove = Vec::new();
        let mut nodes_to_add = Vec::new();

        // Find all Gemv operations and group by input tensor
        for node in &dag.nodes {
            if let Operation::Gemv { input, .. } = &node.op {
                if nodes_to_remove.contains(&node.id) {
                    continue;
                }

                // Find all other Gemv nodes with the same input
                let same_input: Vec<_> = dag
                    .nodes
                    .iter()
                    .filter(|n| {
                        if let Operation::Gemv { input: inp, .. } = &n.op {
                            *inp == *input && !nodes_to_remove.contains(&n.id)
                        } else {
                            false
                        }
                    })
                    .collect();

                // If we have exactly 3 projections (Q, K, V), fuse them
                if same_input.len() == 3 {
                    // Sort by ID to get consistent ordering (Q, K, V)
                    let mut projections: Vec<_> =
                        same_input.into_iter().collect();
                    projections.sort_by_key(|n| n.id);

                    // Extract projection info
                    let get_info = |node: &DagNode| {
                        if let Operation::Gemv {
                            weight,
                            output,
                            out_size,
                            ..
                        } = &node.op
                        {
                            Some((*weight, *output, *out_size))
                        } else {
                            None
                        }
                    };

                    if let (
                        Some((q_weight, q_out, q_size)),
                        Some((k_weight, k_out, k_size)),
                        Some((v_weight, v_out, v_size)),
                    ) = (
                        get_info(projections[0]),
                        get_info(projections[1]),
                        get_info(projections[2]),
                    ) {
                        // Mark nodes for removal
                        let ids: Vec<usize> =
                            projections.iter().map(|n| n.id).collect();
                        nodes_to_remove.extend(ids.clone());

                        // Combine dependencies
                        let mut combined_deps = Vec::new();
                        for proj in &projections {
                            for &dep in &proj.dependencies {
                                if !combined_deps.contains(&dep)
                                    && !nodes_to_remove.contains(&dep)
                                {
                                    combined_deps.push(dep);
                                }
                            }
                        }

                        let fused_node = DagNode {
                            id: dag.nodes.len() + nodes_to_add.len(),
                            op: Operation::FusedQKV {
                                input: *input,
                                q_weight,
                                k_weight,
                                v_weight,
                                output_q: q_out,
                                output_k: k_out,
                                output_v: v_out,
                                q_out_size: q_size,
                                k_out_size: k_size,
                                v_out_size: v_size,
                            },
                            dependencies: combined_deps,
                            device_preference: DevicePreference::Gpu,
                        };

                        nodes_to_add.push(fused_node);
                        fused_count += 3; // Three nodes fused into one
                    }
                }
            }
        }

        Self::apply_modifications(dag, nodes_to_remove, nodes_to_add);
        fused_count
    }

    /// Fuse gate and up projections (both use ffn_norm output)
    /// Similar to QKV fusion but for 2 projections
    fn fuse_gate_up_projections(dag: &mut ComputeDag) -> usize {
        let mut fused_count = 0;
        let mut nodes_to_remove = Vec::new();
        let mut nodes_to_add = Vec::new();

        // Find pairs of Gemv operations that use the same input
        for node in &dag.nodes {
            if let Operation::Gemv { input, weight, output, out_size } =
                &node.op
            {
                if nodes_to_remove.contains(&node.id) {
                    continue;
                }

                // Find another Gemv with the same input
                for other in &dag.nodes {
                    if other.id <= node.id
                        || nodes_to_remove.contains(&other.id)
                    {
                        continue;
                    }
                    if let Operation::Gemv { input: other_input, .. } =
                        &other.op
                    {
                        if *other_input == *input {
                            // Found a pair with same input - fuse them
                            let other_op = &other.op;
                            if let Operation::Gemv {
                                weight: other_weight,
                                output: other_output,
                                out_size: other_out_size,
                                ..
                            } = other_op
                            {
                                nodes_to_remove.push(node.id);
                                nodes_to_remove.push(other.id);

                                // Combine dependencies
                                let mut combined_deps = Vec::new();
                                for &dep in &node.dependencies {
                                    if !nodes_to_remove.contains(&dep) {
                                        combined_deps.push(dep);
                                    }
                                }
                                for &dep in &other.dependencies {
                                    if !combined_deps.contains(&dep)
                                        && !nodes_to_remove.contains(&dep)
                                    {
                                        combined_deps.push(dep);
                                    }
                                }

                                let fused_node = DagNode {
                                    id: dag.nodes.len() + nodes_to_add.len(),
                                    op: Operation::FusedGateUp {
                                        input: *input,
                                        gate_weight: *weight,
                                        up_weight: *other_weight,
                                        output_gate: *output,
                                        output_up: *other_output,
                                        intermediate_size: *out_size,
                                        hidden_size: 0,
                                    },
                                    dependencies: combined_deps,
                                    device_preference: DevicePreference::Gpu,
                                };

                                nodes_to_add.push(fused_node);
                                fused_count += 2;
                            }
                            break;
                        }
                    }
                }
            }
        }

        Self::apply_modifications(dag, nodes_to_remove, nodes_to_add);
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

    /// Fuse Add + RMSNorm → FusedAddNorm
    /// Pattern: residual = Add(layer_input, wo_output); ffn_norm = RMSNorm(residual, weight)
    /// This is from vLLM's fused_add_rms_norm kernel
    fn fuse_add_norm(dag: &mut ComputeDag) -> usize {
        let mut fused_count = 0;
        let mut nodes_to_remove = Vec::new();
        let mut nodes_to_add = Vec::new();

        for add_node in &dag.nodes {
            if let Operation::Add { lhs, rhs, output: add_output } =
                &add_node.op
            {
                if nodes_to_remove.contains(&add_node.id) {
                    continue;
                }

                // Find RmsNorm that uses this Add output as input
                for norm_node in &dag.nodes {
                    if norm_node.id <= add_node.id
                        || nodes_to_remove.contains(&norm_node.id)
                    {
                        continue;
                    }
                    if let Operation::RmsNorm {
                        input: norm_input,
                        weight: norm_weight,
                        output: norm_output,
                        epsilon,
                    } = &norm_node.op
                    {
                        if *norm_input == *add_output {
                            // Found Add + RmsNorm pattern
                            nodes_to_remove.push(add_node.id);
                            nodes_to_remove.push(norm_node.id);

                            // Combine dependencies
                            let mut combined_deps = Vec::new();
                            for &dep in &add_node.dependencies {
                                if !nodes_to_remove.contains(&dep) {
                                    combined_deps.push(dep);
                                }
                            }
                            for &dep in &norm_node.dependencies {
                                if !combined_deps.contains(&dep)
                                    && !nodes_to_remove.contains(&dep)
                                {
                                    combined_deps.push(dep);
                                }
                            }

                            let fused_node = DagNode {
                                id: dag.nodes.len() + nodes_to_add.len(),
                                op: Operation::FusedAddNorm {
                                    lhs: *lhs,
                                    rhs: *rhs,
                                    norm_weight: *norm_weight,
                                    output: *norm_output,
                                    epsilon: *epsilon,
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
