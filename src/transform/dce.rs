// src/transform/dce.rs

use std::collections::{HashMap, HashSet};
use crate::ir::dag::DagGraph;

pub struct DCEPass;

impl DCEPass {
    pub fn apply(&self, graph: &mut DagGraph) -> bool {
        let mut changed = false;

        // 1. 标记所有需要保留的节点（从输出反向遍历）
        let mut live_nodes: HashSet<u64> = HashSet::new();
        let mut worklist: Vec<u64> = graph.outputs.clone();

        while let Some(value_id) = worklist.pop() {
            // 找到产生这个值的算子
            let producer = graph.values.get(&value_id)
                .and_then(|v| v.producer);

            if let Some(op_id) = producer {
                if !live_nodes.contains(&op_id) {
                    live_nodes.insert(op_id);
                    // 将该算子的所有输入加入工作列表
                    if let Some(op) = graph.ops.get(&op_id) {
                        for &in_id in &op.inputs {
                            worklist.push(in_id);
                        }
                    }
                }
            }
        }

        // 2. 删除不在 live_nodes 中的算子
        let dead_ops: Vec<u64> = graph.ops.keys()
            .filter(|&id| !live_nodes.contains(id))
            .cloned()
            .collect();

        if !dead_ops.is_empty() {
            changed = true;
            for id in dead_ops {
                graph.ops.remove(&id);
            }
        }

        // 3. 删除没有 producer 且不在输入/输出中的 Value
        let live_values: HashSet<u64> = graph.outputs.iter().cloned().collect();
        let dead_values: Vec<u64> = graph.values.keys()
            .filter(|&id| {
                // 保留输入、输出、以及被 live 算子使用的值
                if graph.inputs.contains(id) || graph.outputs.contains(id) {
                    return false;
                }
                // 检查是否有 live 算子使用这个值
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
