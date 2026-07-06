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

            // 从 inputs[1] 提取索引数据（假设是 i64）
            // 需要将 Tensor 的数据转为 Vec<i64>
            // 目前 Tensor 只能存 f32 或 i8，需要支持 i64
            // 暂时用 f32 转 i64
            let indices_f32 = inputs[1].data_as_f32();
            let indices: Vec<i64> = indices_f32.iter().map(|&x| x as i64).collect();
            let indices_shape = inputs[1].shape();

            let result = inputs[0].gather(&indices, &indices_shape, dim);
            Ok(vec![result])
        }
        "scatter" => {
            if inputs.len() < 3 {
                return Err("scatter requires 3 inputs (input, indices, src)".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as i32), _ => None })
                .unwrap_or(0);

            // 从 inputs[1] 提取索引数据
            let indices_f32 = inputs[1].data_as_f32();
            let indices: Vec<i64> = indices_f32.iter().map(|&x| x as i64).collect();
            let indices_shape = inputs[1].shape();

            let result = inputs[0].scatter(&indices, &indices_shape, &inputs[2], dim);
            Ok(vec![result])
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

            let (values_tensor, indices_tensor) = inputs[0].topk(k, dim, largest);

            Ok(vec![values_tensor, indices_tensor])
        }
        "argmax" => {
            if inputs.is_empty() {
                return Err("argmax requires 1 input".to_string());
            }
            let result = inputs[0].argmax();
            // 把 Vec<i64> 转成 Tensor
            let shape = vec![result.len()];
            let data_f32: Vec<f32> = result.iter().map(|&x| x as f32).collect();
            let tensor = Tensor::new_f32(&data_f32, &shape);
            Ok(vec![tensor])
        }
        "argmin" => {
            if inputs.is_empty() {
                return Err("argmin requires 1 input".to_string());
            }
            let result = inputs[0].argmin();
            let shape = vec![result.len()];
            let data_f32: Vec<f32> = result.iter().map(|&x| x as f32).collect();
            let tensor = Tensor::new_f32(&data_f32, &shape);
            Ok(vec![tensor])
        }
        _ => Err(format!("Unknown index op: {}", op_type)),
    }
}
