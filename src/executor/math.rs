// src/executor/math.rs

use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_math(
    op_type: &str,
    inputs: &[Tensor<f32>],
    _attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    match op_type {
        "add" => {
            if inputs.len() < 2 {
                return Err("add requires 2 inputs".to_string());
            }
            let result = crate::ops::math::add::add(&inputs[0], &inputs[1]);
            Ok(vec![result])
        }
        "sub" => {
            if inputs.len() < 2 {
                return Err("sub requires 2 inputs".to_string());
            }
            let result = crate::ops::math::sub::sub(&inputs[0], &inputs[1]);
            Ok(vec![result])
        }
        "mul" => {
            if inputs.len() < 2 {
                return Err("mul requires 2 inputs".to_string());
            }
            let result = crate::ops::math::mul::mul(&inputs[0], &inputs[1]);
            Ok(vec![result])
        }
        "div" => {
            if inputs.len() < 2 {
                return Err("div requires 2 inputs".to_string());
            }
            let result = crate::ops::math::div::div(&inputs[0], &inputs[1]);
            Ok(vec![result])
        }
        "matmul" => {
            if inputs.len() < 2 {
                return Err("matmul requires 2 inputs".to_string());
            }
            let result = crate::ops::linalg::matmul::matmul(&inputs[0], &inputs[1]);
            Ok(vec![result])
        }
        "pow" => {
            if inputs.len() < 2 {
                return Err("pow requires 2 inputs".to_string());
            }
            let result = crate::ops::math::pow::pow(&inputs[0], &inputs[1]);
            Ok(vec![result])
        }
        "exp" => {
            if inputs.is_empty() {
                return Err("exp requires 1 input".to_string());
            }
            let result = crate::ops::math::exp::exp(&inputs[0]);
            Ok(vec![result])
        }
        "sqrt" => {
            if inputs.is_empty() {
                return Err("sqrt requires 1 input".to_string());
            }
            let result = crate::ops::math::sqrt::sqrt(&inputs[0]);
            Ok(vec![result])
        }
        "log" => {
            if inputs.is_empty() {
                return Err("log requires 1 input".to_string());
            }
            let result = crate::ops::math::log::log(&inputs[0]);
            Ok(vec![result])
        }
        _ => Err(format!("Unknown math op: {}", op_type)),
    }
}
