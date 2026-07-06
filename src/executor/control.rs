// src/executor/control.rs

use std::collections::HashMap;
use crate::ffi::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_control(
    op_type: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor>, String> {
    match op_type {
        "where" => {
            if inputs.len() < 3 {
                return Err("where requires 3 inputs (condition, true_val, false_val)".to_string());
            }

            // 从 inputs[0] 提取条件数据
            let condition: Vec<u8> = if inputs[0].is_quantized() {
                inputs[0].data_as_i8().iter().map(|&x| if x != 0 { 1 } else { 0 }).collect()
            } else {
                inputs[0].data_as_f32().iter().map(|&x| if x != 0.0 { 1 } else { 0 }).collect()
            };
            let condition_shape = inputs[0].shape();

            // 关联函数调用
            let result = Tensor::where_(&condition, &condition_shape, &inputs[1], &inputs[2]);
            Ok(vec![result])
        }
        "sort" => {
            if inputs.is_empty() {
                return Err("sort requires 1 input".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(-1);
            let ascending = attrs.get("ascending")
                .and_then(|v| match v { AttrValue::Bool(b) => Some(*b), _ => None })
                .unwrap_or(true);

            let (values_tensor, indices_tensor) = inputs[0].sort(dim, ascending);

            Ok(vec![values_tensor, indices_tensor])
        }
        _ => Err(format!("Unknown control op: {}", op_type)),
    }
}
