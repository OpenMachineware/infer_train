// src/executor/math.rs

use std::collections::HashMap;
use crate::ffi::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_math(
    op_type: &str,
    inputs: &[Tensor],
    _attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor>, String> {
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
        "pow" => {
            if inputs.len() < 2 {
                return Err("pow requires 2 inputs".to_string());
            }
            let result = inputs[0].pow(&inputs[1]);
            Ok(vec![result])
        }
        "exp" => {
            if inputs.is_empty() {
                return Err("exp requires 1 input".to_string());
            }
            let result = inputs[0].exp();
            Ok(vec![result])
        }
        "sqrt" => {
            if inputs.is_empty() {
                return Err("sqrt requires 1 input".to_string());
            }
            let result = inputs[0].sqrt();
            Ok(vec![result])
        }
        "log" => {
            if inputs.is_empty() {
                return Err("log requires 1 input".to_string());
            }
            let result = inputs[0].log();
            Ok(vec![result])
        }
        _ => Err(format!("Unknown math op: {}", op_type)),
    }
}
