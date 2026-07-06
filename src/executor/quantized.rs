use std::collections::HashMap;
use crate::ffi::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_quantized(
    op_type: &str,
    inputs: &[Tensor],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor>, String> {
    match op_type {
        "quantized_add" => {
            if inputs.len() < 2 {
                return Err("quantized_add requires 2 inputs".to_string());
            }
            let result = inputs[0].quantized_add(&inputs[1]);
            Ok(vec![result])
        }
        "quantized_sub" => {
            if inputs.len() < 2 {
                return Err("quantized_sub requires 2 inputs".to_string());
            }
            let result = inputs[0].quantized_sub(&inputs[1]);
            Ok(vec![result])
        }
        "quantized_mul" => {
            if inputs.len() < 2 {
                return Err("quantized_mul requires 2 inputs".to_string());
            }
            let result = inputs[0].quantized_mul(&inputs[1]);
            Ok(vec![result])
        }
        "quantized_div" => {
            if inputs.len() < 2 {
                return Err("quantized_div requires 2 inputs".to_string());
            }
            let result = inputs[0].quantized_div(&inputs[1]);
            Ok(vec![result])
        }
        "quantized_matmul" => {
            if inputs.len() < 2 {
                return Err("quantized_matmul requires 2 inputs".to_string());
            }
            let result = inputs[0].quantized_matmul(&inputs[1]);
            Ok(vec![result])
        }
        "quantized_relu" => {
            if inputs.is_empty() {
                return Err("quantized_relu requires 1 input".to_string());
            }
            let result = inputs[0].quantized_relu();
            Ok(vec![result])
        }
        "quantized_sigmoid" => {
            if inputs.is_empty() {
                return Err("quantized_sigmoid requires 1 input".to_string());
            }
            let result = inputs[0].quantized_sigmoid();
            Ok(vec![result])
        }
        "quantized_conv2d" => {
            if inputs.len() < 2 {
                return Err("quantized_conv2d requires at least 2 inputs".to_string());
            }
            let stride = get_int(attrs, "stride", 1);
            let padding = get_int(attrs, "padding", 0);
            let dilation = get_int(attrs, "dilation", 1);
            let groups = get_int(attrs, "groups", 1);
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            let result = inputs[0].quantized_conv2d(&inputs[1], bias, stride, padding, dilation, groups);
            Ok(vec![result])
        }
        "quantized_linear" => {
            if inputs.len() < 2 {
                return Err("quantized_linear requires at least 2 inputs".to_string());
            }
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            let result = inputs[0].quantized_linear(&inputs[1], bias);
            Ok(vec![result])
        }
        "quantized_exp" => {
            if inputs.is_empty() {
                return Err("quantized_exp requires 1 input".to_string());
            }
            let result = inputs[0].quantized_exp();
            Ok(vec![result])
        }
        "quantized_sqrt" => {
            if inputs.is_empty() {
                return Err("quantized_sqrt requires 1 input".to_string());
            }
            let result = inputs[0].quantized_sqrt();
            Ok(vec![result])
        }
        "quantized_abs" => {
            if inputs.is_empty() {
                return Err("quantized_abs requires 1 input".to_string());
            }
            let result = inputs[0].quantized_abs();
            Ok(vec![result])
        }
        "quantized_neg" => {
            if inputs.is_empty() {
                return Err("quantized_neg requires 1 input".to_string());
            }
            let result = inputs[0].quantized_neg();
            Ok(vec![result])
        }
        "quantized_clamp" => {
            if inputs.is_empty() {
                return Err("quantized_clamp requires 1 input".to_string());
            }
            let min_val = attrs.get("min_val")
                .and_then(|v| match v { AttrValue::Float(f) => Some(*f as f32), _ => None })
                .unwrap_or(0.0);
            let max_val = attrs.get("max_val")
                .and_then(|v| match v { AttrValue::Float(f) => Some(*f as f32), _ => None })
                .unwrap_or(1.0);
            let result = inputs[0].quantized_clamp(min_val, max_val);
            Ok(vec![result])
        }
        "quantized_maxpool2d" => {
            if inputs.is_empty() {
                return Err("quantized_maxpool2d requires 1 input".to_string());
            }
            let kernel_size = get_int(attrs, "kernel_size", 2);
            let stride = get_int(attrs, "stride", kernel_size);
            let padding = get_int(attrs, "padding", 0);
            let result = inputs[0].quantized_maxpool2d(kernel_size, stride, padding);
            Ok(vec![result])
        }
        "quantized_avgpool2d" => {
            if inputs.is_empty() {
                return Err("quantized_avgpool2d requires 1 input".to_string());
            }
            let kernel_size = get_int(attrs, "kernel_size", 2);
            let stride = get_int(attrs, "stride", kernel_size);
            let padding = get_int(attrs, "padding", 0);
            let result = inputs[0].quantized_avgpool2d(kernel_size, stride, padding);
            Ok(vec![result])
        }
        "quantized_batchnorm2d" => {
            if inputs.len() < 5 {
                return Err("quantized_batchnorm2d requires 5 inputs".to_string());
            }
            let eps = get_float(attrs, "eps", 1e-5);
            let result = inputs[0].quantized_batchnorm2d(
                &inputs[1], &inputs[2], &inputs[3], &inputs[4], eps
            );
            Ok(vec![result])
        }
        _ => Err(format!("Unknown quantized op: {}", op_type)),
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
