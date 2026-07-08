// src/ir/cfg.rs

use crate::ir::dag::{AttrValue, DataType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgOp {
    pub id: u64,
    pub op_type: String,
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
    pub attrs: HashMap<String, AttrValue>,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgBlock {
    pub id: u64,
    pub name: String,
    pub ops: Vec<CfgOp>,
    pub successors: Vec<u64>,
    pub predecessors: Vec<u64>,
    pub is_entry: bool,
    pub is_exit: bool,
    // 分支信息
    pub branch_info: Option<BranchInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    pub condition_value: u64, // 条件值的 ID
    pub true_branch: u64,     // true 分支的块 ID
    pub false_branch: u64,    // false 分支的块 ID
    pub merge_block: u64,     // 合并块 ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgGraph {
    pub name: String,
    pub blocks: HashMap<u64, CfgBlock>,
    pub entry: u64,
    pub exits: Vec<u64>,
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
    pub next_id: u64,
    // 值映射 (value_id -> (type, shape))
    pub value_types: HashMap<u64, (DataType, Vec<i64>)>,
}

impl CfgGraph {
    pub fn new(name: &str) -> Self {
        CfgGraph {
            name: name.to_string(),
            blocks: HashMap::new(),
            entry: 0,
            exits: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
            value_types: HashMap::new(),
        }
    }

    pub fn add_block(&mut self, name: &str) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.blocks.insert(
            id,
            CfgBlock {
                id,
                name: name.to_string(),
                ops: Vec::new(),
                successors: Vec::new(),
                predecessors: Vec::new(),
                is_entry: false,
                is_exit: false,
                branch_info: None,
            },
        );
        id
    }

    pub fn set_entry(&mut self, id: u64) {
        self.entry = id;
        if let Some(block) = self.blocks.get_mut(&id) {
            block.is_entry = true;
        }
    }

    pub fn add_op(&mut self, block_id: u64, op: CfgOp) -> Result<(), String> {
        let block = self
            .blocks
            .get_mut(&block_id)
            .ok_or_else(|| format!("Block {} not found", block_id))?;
        block.ops.push(op);
        Ok(())
    }

    pub fn add_edge(&mut self, from: u64, to: u64) -> Result<(), String> {
        if let Some(block) = self.blocks.get_mut(&from) {
            if !block.successors.contains(&to) {
                block.successors.push(to);
            }
        }
        if let Some(block) = self.blocks.get_mut(&to) {
            if !block.predecessors.contains(&from) {
                block.predecessors.push(from);
            }
        }
        Ok(())
    }

    pub fn set_branch(
        &mut self,
        block_id: u64,
        condition: u64,
        true_branch: u64,
        false_branch: u64,
        merge: u64,
    ) -> Result<(), String> {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.branch_info = Some(BranchInfo {
                condition_value: condition,
                true_branch,
                false_branch,
                merge_block: merge,
            });
        }
        Ok(())
    }

    pub fn add_input(
        &mut self,
        value_id: u64,
        dtype: DataType,
        shape: Vec<i64>,
    ) {
        self.inputs.push(value_id);
        self.value_types.insert(value_id, (dtype, shape));
    }

    pub fn add_output(&mut self, value_id: u64) {
        self.outputs.push(value_id);
        // 标记为输出块
        for block in self.blocks.values_mut() {
            if block.ops.iter().any(|op| op.outputs.contains(&value_id)) {
                block.is_exit = true;
                self.exits.push(block.id);
            }
        }
    }

    pub fn topological_sort(&self) -> Result<Vec<u64>, String> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![self.entry];

        while let Some(block_id) = stack.pop() {
            if visited.contains(&block_id) {
                continue;
            }
            visited.insert(block_id);

            if let Some(block) = self.blocks.get(&block_id) {
                result.push(block_id);
                for &succ in &block.successors {
                    if !visited.contains(&succ) {
                        stack.push(succ);
                    }
                }
            }
        }

        // 检查是否所有块都被访问
        if result.len() != self.blocks.len() {
            // 处理未访问的块（可能是孤立块）
            for (&id, _) in &self.blocks {
                if !visited.contains(&id) {
                    result.push(id);
                }
            }
        }

        Ok(result)
    }
}
