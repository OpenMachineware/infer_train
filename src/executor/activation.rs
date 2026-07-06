// src/executor/activation.rs

use std::collections::HashMap;
use crate::ffi::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_activation(
    op_type: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor>, String> {
    match op_type {
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
        "gelu" => {
            if inputs.is_empty() {
                return Err("gelu requires 1 input".to_string());
            }
            let result = inputs[0].gelu();
            Ok(vec![result])
        }
        "silu" => {
            if inputs.is_empty() {
                return Err("silu requires 1 input".to_string());
            }
            let result = inputs[0].silu();
            Ok(vec![result])
        }
        "softmax" => {
            if inputs.is_empty() {
                return Err("softmax requires 1 input".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as i32),
                    _ => None,
                })
                .unwrap_or(-1);
            let result = inputs[0].softmax(dim);
            Ok(vec![result])
        }
        "leaky_relu" => {
            if inputs.is_empty() {
                return Err("leaky_relu requires 1 input".to_string());
            }
            let alpha = attrs.get("alpha")
                .and_then(|v| match v {
                    AttrValue::Float(f) => Some(*f as f32),
                    AttrValue::Int(i) => Some(*i as f32),
                    _ => None,
                })
                .unwrap_or(0.01);
            let result = inputs[0].leaky_relu(alpha);
            Ok(vec![result])
        }
        "elu" => {
            if inputs.is_empty() {
                return Err("elu requires 1 input".to_string());
            }
            let alpha = attrs.get("alpha")
                .and_then(|v| match v {
                    AttrValue::Float(f) => Some(*f as f32),
                    AttrValue::Int(i) => Some(*i as f32),
                    _ => None,
                })
                .unwrap_or(1.0);
            let result = inputs[0].elu(alpha);
            Ok(vec![result])
        }
        "relu6" => {
            if inputs.is_empty() {
                return Err("relu6 requires 1 input".to_string());
            }
            let result = inputs[0].relu6();
            Ok(vec![result])
        }
        "log_softmax" => {
            if inputs.is_empty() {
                return Err("log_softmax requires 1 input".to_string());
            }
            let dim = attrs.get("dim")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as i32),
                    _ => None,
                })
                .unwrap_or(-1);
            let result = inputs[0].log_softmax(dim);
            Ok(vec![result])
        }
        _ => Err(format!("Unknown activation op: {}", op_type)),
    }
}
