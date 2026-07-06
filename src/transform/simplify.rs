// src/transform/simplify.rs

use std::collections::HashMap;
use crate::ir::dag::{DagGraph, DataType};

pub struct AlgebraicSimplifyPass;

impl AlgebraicSimplifyPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut replacements: HashMap<u64, u64> = HashMap::new();

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            // ✅ 先取出 op 的数据，释放借用
            let op_data = match graph.ops.get(&op_id) {
                Some(op) => {
                    // 复制需要的数据到局部变量
                    Some((
                        op_id,
                        op.op_type.clone(),
                        op.inputs.clone(),
                        op.outputs.clone(),
                    ))
                }
                None => continue,
            };

            let (_, op_type, inputs, outputs) = match op_data {
                Some(data) => data,
                None => continue,
            };

            // 检查是否所有输入都是常量
            let all_constant = inputs.iter().all(|&id| graph.constants.contains_key(&id));
            if all_constant && !inputs.is_empty() {
                continue;
            }

            match op_type.as_str() {
                "add" => {
                    if inputs.len() >= 2 {
                        let in1 = inputs[0];
                        let in2 = inputs[1];

                        if graph.is_zero_constant(in2) {
                            replacements.insert(outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_zero_constant(in1) {
                            replacements.insert(outputs[0], in2);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "sub" => {
                    if inputs.len() >= 2 {
                        let in1 = inputs[0];
                        let in2 = inputs[1];

                        if graph.is_zero_constant(in2) {
                            replacements.insert(outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if in1 == in2 {
                            // ✅ 先获取 dtype，再调用 mutable 方法
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "mul" => {
                    if inputs.len() >= 2 {
                        let in1 = inputs[0];
                        let in2 = inputs[1];

                        if graph.is_one_constant(in2) {
                            replacements.insert(outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_one_constant(in1) {
                            replacements.insert(outputs[0], in2);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_zero_constant(in2) || graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "div" => {
                    if inputs.len() >= 2 {
                        let in1 = inputs[0];
                        let in2 = inputs[1];

                        if graph.is_one_constant(in2) {
                            replacements.insert(outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "pow" => {
                    if inputs.len() >= 2 {
                        let in1 = inputs[0];
                        let in2 = inputs[1];

                        if graph.is_one_constant(in2) {
                            replacements.insert(outputs[0], in1);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_one_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let one_id = graph.get_or_create_one(dtype);
                            replacements.insert(outputs[0], one_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "exp" => {
                    if inputs.len() >= 1 {
                        let in1 = inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let one_id = graph.get_or_create_one(dtype);
                            replacements.insert(outputs[0], one_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "log" => {
                    if inputs.len() >= 1 {
                        let in1 = inputs[0];
                        if graph.is_one_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "sqrt" => {
                    if inputs.len() >= 1 {
                        let in1 = inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                        if graph.is_one_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let one_id = graph.get_or_create_one(dtype);
                            replacements.insert(outputs[0], one_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "abs" => {
                    if inputs.len() >= 1 {
                        let in1 = inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                "neg" => {
                    if inputs.len() >= 1 {
                        let in1 = inputs[0];
                        if graph.is_zero_constant(in1) {
                            let dtype = graph.values.get(&outputs[0])
                                .map(|v| v.ty.dtype)
                                .unwrap_or(DataType::F32);
                            let zero_id = graph.get_or_create_zero(dtype);
                            replacements.insert(outputs[0], zero_id);
                            to_remove.push(op_id);
                            changed = true;
                            continue;
                        }
                    }
                }
                _ => {}
            }
        }

        // 应用替换
        for (old_id, new_id) in &replacements {
            Self::apply_replacement(graph, *old_id, *new_id);
        }

        for id in to_remove {
            graph.ops.remove(&id);
        }

        changed
    }

    fn apply_replacement(graph: &mut DagGraph, old_id: u64, new_id: u64) {
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

        for output in &mut graph.outputs {
            if *output == old_id {
                *output = new_id;
            }
        }

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

        if let Some(data) = graph.constants.get(&old_id) {
            if !graph.constants.contains_key(&new_id) {
                graph.constants.insert(new_id, data.clone());
            }
        }
    }
}
