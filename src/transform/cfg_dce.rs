// src/transform/cfg_dce.rs

use std::collections::{HashSet, VecDeque};
use crate::ir::cfg::CfgGraph;

pub struct CfgDCEPass;

impl CfgDCEPass {
    pub fn apply(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;

        // 1. 标记所有可达的基本块
        let reachable = Self::mark_reachable_blocks(cfg);

        // 2. 删除不可达的基本块
        let dead_blocks: Vec<u64> = cfg.blocks.keys()
            .filter(|&id| !reachable.contains(id))
            .cloned()
            .collect();

        if !dead_blocks.is_empty() {
            changed = true;
            for id in dead_blocks {
                cfg.blocks.remove(&id);
                // 从其他块的 successors/predecessors 中移除
                for block in cfg.blocks.values_mut() {
                    block.successors.retain(|&x| x != id);
                    block.predecessors.retain(|&x| x != id);
                }
                // 从 exits 中移除
                cfg.exits.retain(|&x| x != id);
            }
        }

        // 3. 删除死算子（没有输出被使用的算子）
        changed |= Self::remove_dead_ops(cfg);

        // 4. 合并只有单个 successor 的块（简化 CFG）
        changed |= Self::merge_single_successor_blocks(cfg);

        changed
    }

    /// 标记所有可达的基本块（从 entry 开始 BFS）
    fn mark_reachable_blocks(cfg: &CfgGraph) -> HashSet<u64> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(cfg.entry);
        reachable.insert(cfg.entry);

        while let Some(block_id) = queue.pop_front() {
            if let Some(block) = cfg.blocks.get(&block_id) {
                for &succ in &block.successors {
                    if !reachable.contains(&succ) {
                        reachable.insert(succ);
                        queue.push_back(succ);
                    }
                }
            }
        }

        reachable
    }

    /// 删除死算子（输出没有在后续被使用）
    fn remove_dead_ops(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;
        let mut used_values = HashSet::new();

        // 1. 收集所有被使用的 value
        for block in cfg.blocks.values() {
            for op in &block.ops {
                for &inp in &op.inputs {
                    used_values.insert(inp);
                }
            }
        }

        // 2. 标记输出被使用的算子
        let mut live_ops = HashSet::new();
        for block in cfg.blocks.values() {
            for op in &block.ops {
                let has_live_output = op.outputs.iter().any(|out| used_values.contains(out));
                if has_live_output {
                    live_ops.insert(op.id);
                }
            }
        }

        // 3. 删除没有 live 输出的算子
        for block in cfg.blocks.values_mut() {
            let len_before = block.ops.len();
            block.ops.retain(|op| live_ops.contains(&op.id));
            if block.ops.len() < len_before {
                changed = true;
            }
        }

        changed
    }

    /// 合并只有单个 successor 的块
    fn merge_single_successor_blocks(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;
        let blocks_to_merge: Vec<u64> = cfg.blocks.iter()
            .filter(|(_, block)| {
                // 块只有一个后继，且不是 exit
                block.successors.len() == 1 &&
                    !block.is_exit &&
                    !block.is_entry
            })
            .map(|(&id, _)| id)
            .collect();

        for &block_id in &blocks_to_merge {
            if let Some(block) = cfg.blocks.get(&block_id) {
                let succ_id = block.successors[0];

                // 将当前块的算子移到后继块
                if let Some(succ_block) = cfg.blocks.get_mut(&succ_id) {
                    let mut ops = block.ops.clone();
                    ops.append(&mut succ_block.ops);
                    succ_block.ops = ops;

                    // 更新前驱
                    for &pred in &block.predecessors {
                        if let Some(pred_block) = cfg.blocks.get_mut(&pred) {
                            for succ in &mut pred_block.successors {
                                if *succ == block_id {
                                    *succ = succ_id;
                                }
                            }
                        }
                    }

                    // 删除当前块
                    cfg.blocks.remove(&block_id);
                    changed = true;
                }
            }
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::CfgGraph;

    #[test]
    fn test_reachable_blocks() {
        let mut cfg = CfgGraph::new("test");
        let entry = cfg.add_block("entry");
        let block1 = cfg.add_block("block1");
        let block2 = cfg.add_block("block2");
        let unreachable = cfg.add_block("unreachable");

        cfg.set_entry(entry);
        cfg.add_edge(entry, block1).unwrap();
        cfg.add_edge(block1, block2).unwrap();
        // unreachable 没有边

        let reachable = CfgDCEPass::mark_reachable_blocks(&cfg);

        assert!(reachable.contains(&entry));
        assert!(reachable.contains(&block1));
        assert!(reachable.contains(&block2));
        assert!(!reachable.contains(&unreachable));
    }
}
