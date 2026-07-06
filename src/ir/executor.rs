// src/ir/executor.rs

use std::collections::HashMap;
use crate::ir::dag::DagGraph;
use crate::ffi::Tensor;

pub struct Executor {
    graph: DagGraph,
    values: HashMap<u64, Tensor>,
}

impl Executor {
    pub fn new(graph: DagGraph) -> Self {
        Executor {
            graph,
            values: HashMap::new(),
        }
    }

    pub fn execute(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.values.clear();

        if inputs.len() != self.graph.inputs.len() {
            return Err(format!(
                "Expected {} inputs, got {}",
                self.graph.inputs.len(),
                inputs.len()
            ));
        }
        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            self.values.insert(input_id, inputs[i].clone());
        }

        let order = self.graph.topological_sort()?;

        // 先收集需要执行的 op_id 列表，避免借用冲突
        let op_ids: Vec<u64> = order;

        for op_id in op_ids {
            // 获取 op 的只读引用
            let op = self.graph.get_op(op_id)
                .ok_or_else(|| format!("Op {} not found", op_id))?
                .clone();  // ← 克隆出来

            self.execute_op(op_id, &op)?;
        }

        let mut result = Vec::new();
        for &out_id in &self.graph.outputs {
            if let Some(t) = self.values.get(&out_id) {
                result.push(t.clone());
            } else {
                return Err(format!("Output value {} not found", out_id));
            }
        }

        Ok(result)
    }

    fn execute_op(&mut self, _op_id: u64, op: &crate::ir::dag::Op) -> Result<(), String> {
        let mut input_tensors = Vec::new();
        for &in_id in &op.inputs {
            if let Some(t) = self.values.get(&in_id) {
                input_tensors.push(t.clone());
            } else {
                return Err(format!("Input value {} not found for op {}", in_id, op.id));
            }
        }

        let outputs = self.dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                self.values.insert(out_id, outputs[i].clone());
            }
        }

        Ok(())
    }

    fn dispatch_op(
        &self,
        op_type: &str,
        inputs: &[Tensor],
        attrs: &HashMap<String, crate::ir::dag::AttrValue>,
    ) -> Result<Vec<Tensor>, String> {
        let get_int = |key: &str, default: i32| -> i32 {
            attrs.get(key)
                .and_then(|v| match v {
                    crate::ir::dag::AttrValue::Int(i) => Some(*i as i32),
                    _ => None,
                })
                .unwrap_or(default)
        };

        match op_type {
            "add" => {
                if inputs.len() < 2 {
                    return Err("add requires 2 inputs".to_string());
                }
                let result = inputs[0].add(&inputs[1]);
                Ok(vec![result])
            }
            "sub" => {
                if inputs.len() < 2 {
                    return Err("sub requires 2 inputs".to_string());
                }
                let result = inputs[0].sub(&inputs[1]);
                Ok(vec![result])
            }
            "mul" => {
                if inputs.len() < 2 {
                    return Err("mul requires 2 inputs".to_string());
                }
                let result = inputs[0].mul(&inputs[1]);
                Ok(vec![result])
            }
            "div" => {
                if inputs.len() < 2 {
                    return Err("div requires 2 inputs".to_string());
                }
                let result = inputs[0].div(&inputs[1]);
                Ok(vec![result])
            }
            "matmul" => {
                if inputs.len() < 2 {
                    return Err("matmul requires 2 inputs".to_string());
                }
                let result = inputs[0].matmul(&inputs[1]);
                Ok(vec![result])
            }
            "relu" => {
                if inputs.is_empty() {
                    return Err("relu requires 1 input".to_string());
                }
                let result = inputs[0].relu();
                Ok(vec![result])
            }
            "sigmoid" => {
                if inputs.is_empty() {
                    return Err("sigmoid requires 1 input".to_string());
                }
                let result = inputs[0].sigmoid();
                Ok(vec![result])
            }
            "tanh" => {
                if inputs.is_empty() {
                    return Err("tanh requires 1 input".to_string());
                }
                let result = inputs[0].tanh();
                Ok(vec![result])
            }
            "softmax" => {
                if inputs.is_empty() {
                    return Err("softmax requires 1 input".to_string());
                }
                let dim = get_int("dim", -1);
                let result = inputs[0].softmax(dim);
                Ok(vec![result])
            }
            "conv2d" => {
                if inputs.len() < 2 {
                    return Err("conv2d requires at least 2 inputs (input, weight)".to_string());
                }
                let stride = get_int("stride", 1);
                let padding = get_int("padding", 0);
                let dilation = get_int("dilation", 1);
                let groups = get_int("groups", 1);
                let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
                let result = inputs[0].conv2d(&inputs[1], bias, stride, padding, dilation, groups);
                Ok(vec![result])
            }
            "linear" => {
                if inputs.len() < 2 {
                    return Err("linear requires at least 2 inputs (input, weight)".to_string());
                }
                let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
                let result = inputs[0].linear(&inputs[1], bias);
                Ok(vec![result])
            }
            "reshape" => {
                if inputs.is_empty() {
                    return Err("reshape requires 1 input".to_string());
                }
                let shape_attr = attrs.get("shape");
                if let Some(crate::ir::dag::AttrValue::Shape(shape)) = shape_attr {
                    let new_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                    let result = inputs[0].reshape(&new_shape);
                    Ok(vec![result])
                } else {
                    Err("reshape requires shape attribute".to_string())
                }
            }
            "transpose" => {
                if inputs.is_empty() {
                    return Err("transpose requires 1 input".to_string());
                }
                let result = inputs[0].transpose();
                Ok(vec![result])
            }
            "constant" => {
                Ok(vec![inputs[0].clone()])
            }
            _ => {
                eprintln!("Warning: operator '{}' not implemented, returning input", op_type);
                Ok(vec![inputs[0].clone()])
            }
        }
    }
}
