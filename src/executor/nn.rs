// src/executor/nn.rs

use std::collections::HashMap;
use crate::ffi::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_nn(
    op_type: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor>, String> {
    match op_type {
        "conv2d" => {
            if inputs.len() < 2 {
                return Err("conv2d requires at least 2 inputs".to_string());
            }
            let stride = get_int(attrs, "stride", 1);
            let padding = get_int(attrs, "padding", 0);
            let dilation = get_int(attrs, "dilation", 1);
            let groups = get_int(attrs, "groups", 1);
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            let result = inputs[0].conv2d(&inputs[1], bias, stride, padding, dilation, groups);
            Ok(vec![result])
        }
        "linear" => {
            if inputs.len() < 2 {
                return Err("linear requires at least 2 inputs".to_string());
            }
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            let result = inputs[0].linear(&inputs[1], bias);
            Ok(vec![result])
        }
        "maxpool2d" => {
            if inputs.is_empty() {
                return Err("maxpool2d requires 1 input".to_string());
            }
            let kernel_size = get_int(attrs, "kernel_size", 2);
            let stride = get_int(attrs, "stride", kernel_size);
            let padding = get_int(attrs, "padding", 0);
            let result = inputs[0].maxpool2d(kernel_size, stride, padding);
            Ok(vec![result])
        }
        "avgpool2d" => {
            if inputs.is_empty() {
                return Err("avgpool2d requires 1 input".to_string());
            }
            let kernel_size = get_int(attrs, "kernel_size", 2);
            let stride = get_int(attrs, "stride", kernel_size);
            let padding = get_int(attrs, "padding", 0);
            let result = inputs[0].avgpool2d(kernel_size, stride, padding);
            Ok(vec![result])
        }
        "batchnorm2d" => {
            if inputs.len() < 5 {
                return Err("batchnorm2d requires 5 inputs".to_string());
            }
            let eps = get_float(attrs, "eps", 1e-5);
            let result = inputs[0].batchnorm2d(&inputs[1], &inputs[2], &inputs[3], &inputs[4], eps);
            Ok(vec![result])
        }
        "layernorm" => {
            if inputs.len() < 3 {
                return Err("layernorm requires 3 inputs".to_string());
            }
            let eps = get_float(attrs, "eps", 1e-5);
            let result = inputs[0].layernorm(&inputs[1], &inputs[2], eps);
            Ok(vec![result])
        }
        "dropout" => {
            if inputs.is_empty() {
                return Err("dropout requires 1 input".to_string());
            }
            let p = get_float(attrs, "p", 0.5);
            let training = get_bool(attrs, "training", false);
            let seed = get_int(attrs, "seed", 0) as u32;
            let result = inputs[0].dropout(p, training, seed);
            Ok(vec![result])
        }
        "embedding" => {
            if inputs.len() < 2 {
                return Err("embedding requires 2 inputs (indices, weight)".to_string());
            }
            let padding_idx = get_int(attrs, "padding_idx", -1);

            // inputs[0] 是索引张量，inputs[1] 是权重张量
            let indices_f32 = inputs[0].data_as_f32();
            let indices: Vec<i64> = indices_f32.iter().map(|&x| x as i64).collect();

            let result = inputs[1].embedding(&indices, padding_idx);
            Ok(vec![result])
        }
        _ => Err(format!("Unknown nn op: {}", op_type)),
    }
}

// 辅助函数
fn get_int(attrs: &HashMap<String, AttrValue>, key: &str, default: i32) -> i32 {
    attrs.get(key)
        .and_then(|v| match v {
            AttrValue::Int(i) => Some(*i as i32),
            AttrValue::IntList(list) if !list.is_empty() => Some(list[0] as i32),
            _ => None,
        })
        .unwrap_or(default)
}

fn get_float(attrs: &HashMap<String, AttrValue>, key: &str, default: f32) -> f32 {
    attrs.get(key)
        .and_then(|v| match v {
            AttrValue::Float(f) => Some(*f as f32),
            AttrValue::Int(i) => Some(*i as f32),
            _ => None,
        })
        .unwrap_or(default)
}

fn get_bool(attrs: &HashMap<String, AttrValue>, key: &str, default: bool) -> bool {
    attrs.get(key)
        .and_then(|v| match v {
            AttrValue::Bool(b) => Some(*b),
            AttrValue::Int(i) => Some(*i != 0),
            _ => None,
        })
        .unwrap_or(default)
}
