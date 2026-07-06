// src/transform/cfg_inline.rs

use std::collections::HashMap;
use crate::ir::cfg::{CfgGraph, CfgBlock, CfgOp};

pub struct CfgInlinePass;

impl CfgInlinePass {
    pub fn apply(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;
        let call_ops: Vec<u64> = cfg.blocks.values()
            .flat_map(|block| block.ops.iter())
            .filter(|op| Self::is_call_op(&op.op_type))
            .map(|op| op.id)
            .collect();

        for op_id in call_ops {
            if Self::inline_call(cfg, op_id) {
                changed = true;
            }
        }

        changed
    }

    fn is_call_op(op_type: &str) -> bool {
        matches!(op_type, "call" | "call_function" | "call_method")
    }

    fn inline_call(cfg: &mut CfgGraph, call_op_id: u64) -> bool {
        // 使用 match 或 if let 替代 ? 操作符
        let (block_id, call_op) = match cfg.blocks.iter_mut()
            .find_map(|(id, block)| {
                block.ops.iter()
                    .find(|op| op.id == call_op_id)
                    .map(|op| (*id, op.clone()))
            }) {
            Some(found) => found,
            None => return false,
        };

        // 获取被调用的函数名
        let func_name = match call_op.attrs.get("function_name") {
            Some(crate::ir::dag::AttrValue::String(s)) => s.clone(),
            _ => return false,
        };

        // 查找被调用的函数
        let func_cfg = match Self::find_function(cfg, &func_name) {
            Some(cfg) => cfg,
            None => return false,
        };

        // 获取 block
        let block = match cfg.blocks.get_mut(&block_id) {
            Some(b) => b,
            None => return false,
        };

        // 找到 call_op 的位置
        let pos = match block.ops.iter().position(|op| op.id == call_op_id) {
            Some(p) => p,
            None => return false,
        };

        // 内联函数体
        let mut inline_ops = func_cfg.blocks.values()
            .flat_map(|b| b.ops.clone())
            .collect::<Vec<CfgOp>>();

        // 重映射 Value ID
        let mut value_map = HashMap::new();
        for op in &mut inline_ops {
            op.id = cfg.next_id;
            cfg.next_id += 1;

            // 映射 inputs
            for inp in &mut op.inputs {
                if let Some(&new_id) = value_map.get(inp) {
                    *inp = new_id;
                }
            }

            // 映射 outputs
            for out in &mut op.outputs {
                let new_id = cfg.next_id;
                cfg.next_id += 1;
                value_map.insert(*out, new_id);
                *out = new_id;
            }
        }

        // 替换原始 call_op
        block.ops.splice(pos..pos+1, inline_ops);

        true
    }

    fn find_function(cfg: &CfgGraph, name: &str) -> Option<CfgGraph> {
        // TODO: 从函数注册表中查找
        // 这里返回 None，需要实现函数存储
        // 可以添加一个 HashMap 存储所有函数
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::CfgGraph;
    use crate::ir::dag::AttrValue;

    #[test]
    fn test_inline_detection() {
        let mut cfg = CfgGraph::new("test");
        let block = cfg.add_block("entry");
        cfg.set_entry(block);

        let mut attrs = HashMap::new();
        attrs.insert("function_name".to_string(),
                     AttrValue::String("matmul".to_string()));

        let op = CfgOp {
            id: 0,
            op_type: "call".to_string(),
            inputs: vec![1, 2],
            outputs: vec![3],
            attrs,
            name: "call_matmul".to_string(),
        };
        cfg.add_op(block, op).unwrap();

        assert!(CfgInlinePass::is_call_op("call"));
        assert!(CfgInlinePass::is_call_op("call_function"));
        assert!(!CfgInlinePass::is_call_op("add"));
    }
}
