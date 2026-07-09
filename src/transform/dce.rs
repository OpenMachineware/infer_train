// src/transform/dce.rs

use crate::ir::dag::DagGraph;
use std::collections::HashSet;

pub struct DCEPass;

impl DCEPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;

        // Mark all nodes that need to be preserved
        // (backward traversal from outputs)
        let mut live_nodes: HashSet<u64> = HashSet::new();
        let mut worklist: Vec<u64> = graph.outputs.clone();

        while let Some(value_id) = worklist.pop() {
            // Find the operator that produces this value
            let producer = graph.values.get(&value_id).and_then(|v| v.producer);

            if let Some(op_id) = producer {
                if !live_nodes.contains(&op_id) {
                    live_nodes.insert(op_id);
                    // Add all inputs of this operator to the worklist
                    if let Some(op) = graph.ops.get(&op_id) {
                        for &in_id in &op.inputs {
                            worklist.push(in_id);
                        }
                    }
                }
            }
        }

        // Remove operators not in live_nodes
        let dead_ops: Vec<u64> = graph
            .ops
            .keys()
            .filter(|&id| !live_nodes.contains(id))
            .cloned()
            .collect();

        if !dead_ops.is_empty() {
            changed = true;
            for id in dead_ops {
                graph.ops.remove(&id);
            }
        }

        // Remove Values that have no producer and are not in inputs/outputs
        let _live_values: HashSet<u64> =
            graph.outputs.iter().cloned().collect();
        let dead_values: Vec<u64> = graph
            .values
            .keys()
            .filter(|&id| {
                // Preserve inputs, outputs, and values used by live operators
                if graph.inputs.contains(id) || graph.outputs.contains(id) {
                    return false;
                }
                // Check if any live operator uses this value
                for op in graph.ops.values() {
                    if op.inputs.contains(id) || op.outputs.contains(id) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        if !dead_values.is_empty() {
            changed = true;
            for id in dead_values {
                graph.values.remove(&id);
                graph.constants.remove(&id);
            }
        }

        changed
    }
}
