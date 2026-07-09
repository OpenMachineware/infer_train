// src/transform/cfg_dce.rs

use crate::ir::cfg::CfgGraph;
use std::collections::{HashSet, VecDeque};

pub struct CfgDCEPass;

impl CfgDCEPass {
    pub fn apply(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;

        // 标记可达块
        let reachable = Self::mark_reachable_blocks(cfg);

        // 删除不可达块
        let dead_blocks: Vec<u64> = cfg
            .blocks
            .keys()
            .filter(|&id| !reachable.contains(id))
            .cloned()
            .collect();

        if !dead_blocks.is_empty() {
            changed = true;
            for id in dead_blocks {
                cfg.blocks.remove(&id);
                for block in cfg.blocks.values_mut() {
                    block.successors.retain(|&x| x != id);
                    block.predecessors.retain(|&x| x != id);
                }
                cfg.exits.retain(|&x| x != id);
            }
        }

        // 删除死算子
        changed |= Self::remove_dead_ops(cfg);

        // 修复：合并单后继块
        changed |= Self::merge_single_successor_blocks(cfg);

        changed
    }

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

    fn remove_dead_ops(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;
        let mut used_values = HashSet::new();

        for block in cfg.blocks.values() {
            for op in &block.ops {
                for &inp in &op.inputs {
                    used_values.insert(inp);
                }
            }
        }

        let mut live_ops = HashSet::new();
        for block in cfg.blocks.values() {
            for op in &block.ops {
                let has_live_output =
                    op.outputs.iter().any(|out| used_values.contains(out));
                if has_live_output {
                    live_ops.insert(op.id);
                }
            }
        }

        for block in cfg.blocks.values_mut() {
            let len_before = block.ops.len();
            block.ops.retain(|op| live_ops.contains(&op.id));
            if block.ops.len() < len_before {
                changed = true;
            }
        }

        changed
    }

    fn merge_single_successor_blocks(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;

        // 先收集需要合并的块 ID
        let blocks_to_merge: Vec<u64> = cfg
            .blocks
            .iter()
            .filter(|(_, block)| {
                block.successors.len() == 1 && !block.is_exit && !block.is_entry
            })
            .map(|(&id, _)| id)
            .collect();

        for &block_id in &blocks_to_merge {
            // 检查块是否还存在（可能已被之前合并删除）
            if !cfg.blocks.contains_key(&block_id) {
                continue;
            }

            // 先收集需要的数据，避免同时持有引用
            let (succ_id, predecessors, ops) = {
                let block = match cfg.blocks.get(&block_id) {
                    Some(b) => b,
                    None => continue,
                };
                let succ_id = block.successors[0];
                let predecessors = block.predecessors.clone();
                let ops = block.ops.clone();
                (succ_id, predecessors, ops)
            };

            // 检查后继块是否存在
            if !cfg.blocks.contains_key(&succ_id) {
                continue;
            }

            // 现在可以安全地修改 cfg.blocks
            // 更新前驱块的 successor
            for pred in &predecessors {
                if let Some(pred_block) = cfg.blocks.get_mut(pred) {
                    for succ in &mut pred_block.successors {
                        if *succ == block_id {
                            *succ = succ_id;
                        }
                    }
                }
            }

            // 将当前块的算子移到后继块
            if let Some(succ_block) = cfg.blocks.get_mut(&succ_id) {
                let mut new_ops = ops.clone();
                new_ops.append(&mut succ_block.ops);
                succ_block.ops = new_ops;
                // 更新前驱
                succ_block.predecessors = predecessors.clone();
            }

            // 删除当前块
            cfg.blocks.remove(&block_id);
            changed = true;
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

        let reachable = CfgDCEPass::mark_reachable_blocks(&cfg);

        assert!(reachable.contains(&entry));
        assert!(reachable.contains(&block1));
        assert!(reachable.contains(&block2));
        assert!(!reachable.contains(&unreachable));
    }
}
