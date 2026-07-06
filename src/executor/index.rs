// src/executor/index.rs

use std::collections::HashMap;
use crate::ffi::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_index(
    op_type: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor>, String> {
    match op_type {
        "gather" => {
            if inputs.len() < 2 {
                return Err("gather requires 2 inputs (input, indices)".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(0);
            // 需要 indices 的 shape
            let indices_shape = inputs[1].shape();
            // 目前 gather 需要 indices 作为 Vec<i64>，但这里 inputs[1] 是 Tensor
            // 需要从 Tensor 提取数据
            // TODO: 实现从 Tensor 提取索引数据
            // 暂时返回错误
            Err(format!("gather not fully implemented yet"))
        }
        "scatter" => {
            if inputs.len() < 3 {
                return Err("scatter requires 3 inputs (input, indices, src)".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(0);
            // 需要 indices 的 shape
            let indices_shape = inputs[1].shape();
            // 需要从 inputs[1] 提取索引数据
            // TODO: 实现
            Err(format!("scatter not fully implemented yet"))
        }
        "topk" => {
            if inputs.is_empty() {
                return Err("topk requires 1 input".to_string());
            }
            let k = attrs.get("k")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as usize), _ => None })
                .unwrap_or(1);
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(-1);
            let largest = attrs.get("largest")
                .and_then(|v| match v { AttrValue::Bool(b) => Some(*b), _ => None })
                .unwrap_or(true);
            let (values, indices) = inputs[0].topk(k, dim, largest);
            // 返回两个张量：值和索引
            // 索引需要转成 Tensor
            // TODO: 实现 indices -> Tensor
            Ok(vec![values])  // 暂时只返回值
        }
        "argmax" => {
            if inputs.is_empty() {
                return Err("argmax requires 1 input".to_string());
            }
            let result = inputs[0].argmax();
            // 返回 int64 索引
            // 需要将 Vec<i64> 转成 Tensor
            // TODO: 实现
            Err(format!("argmax not fully implemented yet"))
        }
        "argmin" => {
            if inputs.is_empty() {
                return Err("argmin requires 1 input".to_string());
            }
            Err(format!("argmin not fully implemented yet"))
        }
        _ => Err(format!("Unknown index op: {}", op_type)),
    }
}
