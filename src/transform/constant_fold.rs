// src/transform/constant_fold.rs

use std::collections::HashMap;
use crate::ir::dag::{DagGraph, Op, AttrValue};

pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    pub fn apply(&self, graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();

        // 删除未使用的变量
        // let mut new_constants = HashMap::new();  // ← 删除这行

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            let mut all_constant = true;
            let mut const_inputs = Vec::new();
            for &in_id in &op.inputs {
                if let Some(data) = graph.constants.get(&in_id) {
                    const_inputs.push(data.clone());
                } else {
                    all_constant = false;
                    break;
                }
            }

            if !all_constant || op.inputs.is_empty() {
                continue;
            }

            if let Some(result_data) = self.fold_op(&op.op_type, &const_inputs, &op.attrs) {
                let out_id = op.outputs[0];
                graph.constants.insert(out_id, result_data);
                to_remove.push(op_id);
                changed = true;
            }
        }

        for id in to_remove {
            graph.ops.remove(&id);
        }

        changed
    }

    fn fold_op(
        &self,
        op_type: &str,
        inputs: &[Vec<u8>],
        _attrs: &HashMap<String, AttrValue>,
    ) -> Option<Vec<u8>> {
        // TODO: 完善常量折叠，支持张量数据
        match op_type {
            "add" => {
                if inputs.len() < 2 {
                    return None;
                }
                let a = self.bytes_to_f32(&inputs[0]);
                let b = self.bytes_to_f32(&inputs[1]);
                Some(self.f32_to_bytes(a + b))
            }
            "sub" => {
                if inputs.len() < 2 {
                    return None;
                }
                let a = self.bytes_to_f32(&inputs[0]);
                let b = self.bytes_to_f32(&inputs[1]);
                Some(self.f32_to_bytes(a - b))
            }
            "mul" => {
                if inputs.len() < 2 {
                    return None;
                }
                let a = self.bytes_to_f32(&inputs[0]);
                let b = self.bytes_to_f32(&inputs[1]);
                Some(self.f32_to_bytes(a * b))
            }
            "div" => {
                if inputs.len() < 2 {
                    return None;
                }
                let a = self.bytes_to_f32(&inputs[0]);
                let b = self.bytes_to_f32(&inputs[1]);
                if b == 0.0 {
                    None
                } else {
                    Some(self.f32_to_bytes(a / b))
                }
            }
            "constant" => {
                Some(inputs[0].clone())
            }
            _ => None,
        }
    }

    // TODO: 完善 bytes ↔ f32，支持张量
    fn bytes_to_f32(&self, bytes: &[u8]) -> f32 {
        if bytes.len() >= 4 {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            0.0
        }
    }

    fn f32_to_bytes(&self, val: f32) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }
}
