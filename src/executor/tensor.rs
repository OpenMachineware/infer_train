// src/executor/tensor.rs

use std::collections::HashMap;
use crate::ffi::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_tensor(
    op_type: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor>, String> {
    match op_type {
        "reshape" => {
            if inputs.is_empty() {
                return Err("reshape requires 1 input".to_string());
            }
            let shape_attr = attrs.get("shape");
            if let Some(AttrValue::Shape(shape)) = shape_attr {
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
        "slice" => {
            if inputs.is_empty() {
                return Err("slice requires 1 input".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(0);
            let start = attrs.get("start")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(0);
            let end = attrs.get("end")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(-1);
            let step = attrs.get("step")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(1);
            let result = inputs[0].slice(dim, start, end, step);
            Ok(vec![result])
        }
        "cat" => {
            if inputs.is_empty() {
                return Err("cat requires at least 1 input".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(0);
            // 收集引用
            let tensor_refs: Vec<&Tensor> = inputs.iter().collect();
            let result = Tensor::cat(&tensor_refs, dim);
            Ok(vec![result])
        }
        "cumsum" => {
            if inputs.is_empty() {
                return Err("cumsum requires 1 input".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(-1);
            let result = inputs[0].cumsum(dim);
            Ok(vec![result])
        }
        "cumprod" => {
            if inputs.is_empty() {
                return Err("cumprod requires 1 input".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(-1);
            let result = inputs[0].cumprod(dim);
            Ok(vec![result])
        }
        _ => Err(format!("Unknown tensor op: {}", op_type)),
    }
}
