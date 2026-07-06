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
            // 需要从 condition Tensor 提取数据
            // 当前 condition 是 Tensor，需要转为 Vec<u8>
            // TODO: 实现条件提取
            Err(format!("where not fully implemented yet"))
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
            let (values, indices) = inputs[0].sort(dim, ascending);
            // TODO: indices -> Tensor
            Ok(vec![values])  // 暂时只返回值
        }
        _ => Err(format!("Unknown control op: {}", op_type)),
    }
}
