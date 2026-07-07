// src/executor/index.rs

use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_index(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    match op_type {
        "topk" => {
            if inputs.is_empty() {
                return Err("topk requires 1 input".to_string());
            }
            let k = attrs.get("k")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as usize), _ => None })
                .unwrap_or(1);
            let dim = attrs.get("dim")
                .and_then(|v| match v { AttrValue::Int(i) => Some(*i as usize), _ => None })
                .unwrap_or(0);
            let largest = attrs.get("largest")
                .and_then(|v| match v { AttrValue::Bool(b) => Some(*b), _ => None })
                .unwrap_or(true);

            let (values, _indices) = crate::ops::embedding_lookup::topk::topk(&inputs[0], k, dim, largest);
            Ok(vec![values])
        }
        _ => Err(format!("Unknown index op: {}", op_type)),
    }
}
