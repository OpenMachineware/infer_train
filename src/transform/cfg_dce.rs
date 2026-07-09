// InferTrain - A Unified Inference and Training Engine
//
// Copyright (c) 2026 Jia Liu & InferTrain Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of InferTrain.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::ir::cfg::CfgGraph;
use std::collections::{HashSet, VecDeque};

pub struct CfgDCEPass;

impl CfgDCEPass {
    pub fn apply(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;

        // Mark reachable blocks
        let reachable = Self::mark_reachable_blocks(cfg);

        // Remove unreachable blocks
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

        // Remove dead operators
        changed |= Self::remove_dead_ops(cfg);

        // Fix: merge blocks with single successor
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

        // Collect block IDs to merge first
        let blocks_to_merge: Vec<u64> = cfg
            .blocks
            .iter()
            .filter(|(_, block)| {
                block.successors.len() == 1 && !block.is_exit && !block.is_entry
            })
            .map(|(&id, _)| id)
            .collect();

        for &block_id in &blocks_to_merge {
            // Check if block still exists (might have been merged and deleted)
            if !cfg.blocks.contains_key(&block_id) {
                continue;
            }

            // Collect needed data first to avoid holding references
            // simultaneously
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

            // Check if successor block exists
            if !cfg.blocks.contains_key(&succ_id) {
                continue;
            }

            // Now it's safe to modify cfg.blocks
            // Update predecessor blocks' successors
            for pred in &predecessors {
                if let Some(pred_block) = cfg.blocks.get_mut(pred) {
                    for succ in &mut pred_block.successors {
                        if *succ == block_id {
                            *succ = succ_id;
                        }
                    }
                }
            }

            // Move current block's operators to successor block
            if let Some(succ_block) = cfg.blocks.get_mut(&succ_id) {
                let mut new_ops = ops.clone();
                new_ops.append(&mut succ_block.ops);
                succ_block.ops = new_ops;
                // Update predecessors
                succ_block.predecessors = predecessors.clone();
            }

            // Remove current block
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
