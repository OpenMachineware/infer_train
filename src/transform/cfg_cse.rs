// src/transform/cfg_cse.rs

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use crate::ir::cfg::{CfgGraph, CfgOp};
use crate::ir::dag::AttrValue;

pub struct CfgCSEPass;

impl CfgCSEPass {
    pub fn apply(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;
        let mut hash_to_op: HashMap<String, (u64, u64)> = HashMap::new(); // hash -> (block_id, op_id)
        let mut replacements: HashMap<u64, u64> = HashMap::new(); // old_value -> new_value

        // 1. 按拓扑顺序遍历所有块
        let block_order = match cfg.topological_sort() {
            Ok(order) => order,
            Err(_) => return false,
        };

        for &block_id in &block_order {
            let block = match cfg.blocks.get(&block_id) {
                Some(b) => b,
                None => continue,
            };

            let mut new_ops = Vec::new();

            for op in &block.ops {
                // 只处理纯函数算子
                if Self::is_pure_operator(&op.op_type) {
                    let hash = Self::hash_op(op);

                    if let Some(&(existing_block_id, existing_op_id)) = hash_to_op.get(&hash) {
                        // 找到重复的算子
                        // 将当前算子的输出替换为已有算子的输出
                        if op.outputs.len() == 1 && existing_block_id != block_id {
                            // 找到已有算子的输出
                            if let Some(existing_block) = cfg.blocks.get(&existing_block_id) {
                                if let Some(existing_op) = existing_block.ops.iter()
                                    .find(|o| o.id == existing_op_id)
                                {
                                    if !existing_op.outputs.is_empty() {
                                        let old_out = op.outputs[0];
                                        let new_out = existing_op.outputs[0];
                                        replacements.insert(old_out, new_out);
                                        changed = true;
                                        continue; // 跳过当前算子
                                    }
                                }
                            }
                        }
                    } else {
                        // 记录新算子
                        if !op.outputs.is_empty() {
                            hash_to_op.insert(hash, (block_id, op.id));
                        }
                    }
                }

                // 保留当前算子
                new_ops.push(op.clone());
            }

            // 更新块的算子列表
            if let Some(block) = cfg.blocks.get_mut(&block_id) {
                block.ops = new_ops;
            }
        }

        // 2. 应用值替换
        if !replacements.is_empty() {
            for block in cfg.blocks.values_mut() {
                for op in &mut block.ops {
                    for inp in &mut op.inputs {
                        if let Some(&new_id) = replacements.get(inp) {
                            *inp = new_id;
                        }
                    }
                }
            }
        }

        changed
    }

    /// 判断是否是纯函数算子
    fn is_pure_operator(op_type: &str) -> bool {
        matches!(op_type,
            "add" | "sub" | "mul" | "div" | "pow" |
            "exp" | "sqrt" | "log" | "log2" | "log10" |
            "abs" | "neg" | "floor" | "ceil" | "round" |
            "relu" | "sigmoid" | "tanh" | "gelu" | "silu" |
            "matmul" | "linear" |
            "conv2d" | "maxpool2d" | "avgpool2d" |
            "batchnorm2d" | "layernorm" |
            "reshape" | "transpose" | "slice" | "cat"
        )
    }

    /// 计算算子的哈希
    fn hash_op(op: &CfgOp) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        op.op_type.hash(&mut hasher);
        op.inputs.hash(&mut hasher);
        op.attrs.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::CfgGraph;

    #[test]
    fn test_cse_duplicate() {
        let mut cfg = CfgGraph::new("test");
        let block1 = cfg.add_block("block1");
        let block2 = cfg.add_block("block2");
        cfg.set_entry(block1);
        cfg.add_edge(block1, block2).unwrap();

        // 在两个块中添加相同的算子
        let op1 = CfgOp {
            id: 0,
            op_type: "add".to_string(),
            inputs: vec![1, 2],
            outputs: vec![3],
            attrs: HashMap::new(),
            name: "add1".to_string(),
        };

        let op2 = CfgOp {
            id: 1,
            op_type: "add".to_string(),
            inputs: vec![1, 2],
            outputs: vec![4],
            attrs: HashMap::new(),
            name: "add2".to_string(),
        };

        cfg.add_op(block1, op1).unwrap();
        cfg.add_op(block2, op2).unwrap();

        assert!(CfgCSEPass::apply(&mut cfg));
    }
}
