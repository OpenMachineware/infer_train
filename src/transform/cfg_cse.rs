// src/transform/cfg_cse.rs

use crate::ir::cfg::{CfgGraph, CfgOp};
use crate::ir::dag::AttrValue;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub struct CfgCSEPass;

impl CfgCSEPass {
    pub fn apply(cfg: &mut CfgGraph) -> bool {
        let mut changed = false;
        let mut hash_to_op: HashMap<String, (u64, u64)> = HashMap::new();
        let mut replacements: HashMap<u64, u64> = HashMap::new();

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
                if Self::is_pure_operator(&op.op_type) {
                    let hash = Self::hash_op(op);

                    if let Some(&(existing_block_id, existing_op_id)) =
                        hash_to_op.get(&hash)
                    {
                        if op.outputs.len() == 1
                            && existing_block_id != block_id
                        {
                            if let Some(existing_block) =
                                cfg.blocks.get(&existing_block_id)
                            {
                                if let Some(existing_op) = existing_block
                                    .ops
                                    .iter()
                                    .find(|o| o.id == existing_op_id)
                                {
                                    if !existing_op.outputs.is_empty() {
                                        let old_out = op.outputs[0];
                                        let new_out = existing_op.outputs[0];
                                        replacements.insert(old_out, new_out);
                                        changed = true;
                                        continue;
                                    }
                                }
                            }
                        }
                    } else {
                        if !op.outputs.is_empty() {
                            hash_to_op.insert(hash, (block_id, op.id));
                        }
                    }
                }

                new_ops.push(op.clone());
            }

            if let Some(block) = cfg.blocks.get_mut(&block_id) {
                block.ops = new_ops;
            }
        }

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

    fn is_pure_operator(op_type: &str) -> bool {
        matches!(
            op_type,
            "add"
                | "sub"
                | "mul"
                | "div"
                | "pow"
                | "exp"
                | "sqrt"
                | "log"
                | "log2"
                | "log10"
                | "abs"
                | "neg"
                | "floor"
                | "ceil"
                | "round"
                | "relu"
                | "sigmoid"
                | "tanh"
                | "gelu"
                | "silu"
                | "matmul"
                | "linear"
                | "conv2d"
                | "maxpool2d"
                | "avgpool2d"
                | "batchnorm2d"
                | "layernorm"
                | "reshape"
                | "transpose"
                | "slice"
                | "cat"
        )
    }

    fn hash_op(op: &CfgOp) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // 哈希 op_type
        op.op_type.hash(&mut hasher);

        // 哈希 inputs
        op.inputs.hash(&mut hasher);

        // 手动哈希 attrs
        let mut keys: Vec<&String> = op.attrs.keys().collect();
        keys.sort(); // 保证顺序一致
        for key in keys {
            key.hash(&mut hasher);
            if let Some(value) = op.attrs.get(key) {
                Self::hash_attr(value, &mut hasher);
            }
        }

        format!("{:016x}", hasher.finish())
    }

    fn hash_attr(
        attr: &AttrValue,
        hasher: &mut std::collections::hash_map::DefaultHasher,
    ) {
        match attr {
            AttrValue::Int(i) => i.hash(hasher),
            AttrValue::Float(f) => f.to_bits().hash(hasher),
            AttrValue::Bool(b) => b.hash(hasher),
            AttrValue::String(s) => s.hash(hasher),
            AttrValue::IntList(list) => list.hash(hasher),
            AttrValue::FloatList(list) => {
                for f in list {
                    f.to_bits().hash(hasher);
                }
            }
            AttrValue::Shape(shape) => shape.hash(hasher),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::cfg::CfgGraph;
    use crate::ir::dag::AttrValue;

    #[test]
    fn test_cse_duplicate() {
        let mut cfg = CfgGraph::new("test");
        let block1 = cfg.add_block("block1");
        let block2 = cfg.add_block("block2");
        cfg.set_entry(block1);
        cfg.add_edge(block1, block2).unwrap();

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

    #[test]
    fn test_hash_attrs() {
        let mut attrs = HashMap::new();
        attrs.insert("stride".to_string(), AttrValue::Int(2));
        attrs.insert("padding".to_string(), AttrValue::Int(1));

        let op = CfgOp {
            id: 0,
            op_type: "conv2d".to_string(),
            inputs: vec![1, 2],
            outputs: vec![3],
            attrs: attrs.clone(),
            name: "conv".to_string(),
        };

        let hash1 = CfgCSEPass::hash_op(&op);

        // 相同 attrs 应该产生相同 hash
        let mut attrs2 = HashMap::new();
        attrs2.insert("stride".to_string(), AttrValue::Int(2));
        attrs2.insert("padding".to_string(), AttrValue::Int(1));

        let op2 = CfgOp {
            id: 1,
            op_type: "conv2d".to_string(),
            inputs: vec![1, 2],
            outputs: vec![4],
            attrs: attrs2,
            name: "conv2".to_string(),
        };

        let hash2 = CfgCSEPass::hash_op(&op2);
        assert_eq!(hash1, hash2);
    }
}
