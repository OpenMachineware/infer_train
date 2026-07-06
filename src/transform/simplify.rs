// src/transform/simplify.rs

use std::collections::HashMap;
use crate::ir::dag::{DagGraph, DataType};

pub struct AlgebraicSimplifyPass;

impl AlgebraicSimplifyPass {
    pub fn apply(&self, graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut replacements: HashMap<u64, u64> = HashMap::new();

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            // 跳过所有输入都是常量的（让 constant_fold 处理）
            let all_constant = op.inputs.iter().all(|&id| graph.constants.contains_key(&id));
            if all_constant && !op.inputs.is_empty() {
                continue;
            }

            match op.op_type.as_str() {
                // ============================================================
                // add: x + 0 → x, 0 + x → x
                // ============================================================
                "add" => {
                    if op.inputs.len() >= 2 {
                        let in1 = op.inputs[0];
                        let in2 = op.inputs[1];

                        if graph.is_zero_constant(in2) {
                            replacements.insert(op.outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_zero_constant(in1) {
                            replacements.insert(op.outputs[0], in2);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // sub: x - 0 → x, x - x → 0
                // ============================================================
                "sub" => {
                    if op.inputs.len() >= 2 {
                        let in1 = op.inputs[0];
                        let in2 = op.inputs[1];

                        if graph.is_zero_constant(in2) {
                            replacements.insert(op.outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if in1 == in2 {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(op.outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // mul: x * 1 → x, 1 * x → x, x * 0 → 0, 0 * x → 0
                // ============================================================
                "mul" => {
                    if op.inputs.len() >= 2 {
                        let in1 = op.inputs[0];
                        let in2 = op.inputs[1];

                        if graph.is_one_constant(in2) {
                            replacements.insert(op.outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_one_constant(in1) {
                            replacements.insert(op.outputs[0], in2);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_zero_constant(in2) || graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(op.outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // div: x / 1 → x, 0 / x → 0
                // ============================================================
                "div" => {
                    if op.inputs.len() >= 2 {
                        let in1 = op.inputs[0];
                        let in2 = op.inputs[1];

                        if graph.is_one_constant(in2) {
                            replacements.insert(op.outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(op.outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // pow: x ^ 1 → x, 1 ^ x → 1
                // ============================================================
                "pow" => {
                    if op.inputs.len() >= 2 {
                        let in1 = op.inputs[0];
                        let in2 = op.inputs[1];

                        if graph.is_one_constant(in2) {
                            replacements.insert(op.outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_one_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let one_id = graph.get_or_create_one(dtype);
                            replacements.insert(op.outputs[0], one_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // exp: exp(0) → 1
                // ============================================================
                "exp" => {
                    if op.inputs.len() >= 1 {
                        let in1 = op.inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let one_id = graph.get_or_create_one(dtype);
                            replacements.insert(op.outputs[0], one_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // log: log(1) → 0
                // ============================================================
                "log" => {
                    if op.inputs.len() >= 1 {
                        let in1 = op.inputs[0];
                        if graph.is_one_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(op.outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // sqrt: sqrt(0) → 0, sqrt(1) → 1
                // ============================================================
                "sqrt" => {
                    if op.inputs.len() >= 1 {
                        let in1 = op.inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(op.outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_one_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let one_id = graph.get_or_create_one(dtype);
                            replacements.insert(op.outputs[0], one_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // abs: abs(0) → 0
                // ============================================================
                "abs" => {
                    if op.inputs.len() >= 1 {
                        let in1 = op.inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(op.outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ============================================================
                // neg: neg(0) → 0
                // ============================================================
                "neg" => {
                    if op.inputs.len() >= 1 {
                        let in1 = op.inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&op.outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(op.outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }

                _ => {}
            }
        }

        // ============================================================
        // 应用替换
        // ============================================================
        for (old_id, new_id) in &replacements {
            Self::apply_replacement(graph, *old_id, *new_id);
        }

        for id in to_remove {
            graph.ops.remove(&id);
        }

        changed
    }

    // ============================================================
    // 工具函数：应用替换
    // ============================================================
    fn apply_replacement(graph: &mut DagGraph, old_id: u64, new_id: u64) {
        // 更新所有 op 的 inputs 和 outputs
        for (_, op) in graph.ops.iter_mut() {
            for input in &mut op.inputs {
                if *input == old_id {
                    *input = new_id;
                }
            }
            for output in &mut op.outputs {
                if *output == old_id {
                    *output = new_id;
                }
            }
        }

        // 更新 graph.outputs
        for output in &mut graph.outputs {
            if *output == old_id {
                *output = new_id;
            }
        }

        // 如果 old_id 有 producer，更新 producer 的 outputs
        if let Some(value) = graph.values.get(&old_id) {
            if let Some(producer_id) = value.producer {
                if let Some(op) = graph.ops.get_mut(&producer_id) {
                    for output in &mut op.outputs {
                        if *output == old_id {
                            *output = new_id;
                        }
                    }
                }
            }
        }

        // 如果 old_id 是常量，复制数据到 new_id（如果 new_id 还没有数据）
        if let Some(data) = graph.constants.get(&old_id) {
            if !graph.constants.contains_key(&new_id) {
                graph.constants.insert(new_id, data.clone());
            }
        }
    }
}
