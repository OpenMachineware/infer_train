// src/executor/quantized.rs

use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::ir::dag::AttrValue;

pub fn dispatch_quantized(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    // 量化算子：暂时用 float 版本替代
    // 实际需要 i8 张量支持
    match op_type {
        "quantized_add" => {
            if inputs.len() < 2 {
                return Err("quantized_add requires 2 inputs".to_string());
            }
            let result = crate::ops::math::add::add(&inputs[0], &inputs[1]);
            Ok(vec![result])
        }
        "quantized_relu" => {
            if inputs.is_empty() {
                return Err("quantized_relu requires 1 input".to_string());
            }
            let result = crate::ops::activation::relu::relu(&inputs[0]);
            Ok(vec![result])
        }
        "quantized_conv2d" => {
            if inputs.len() < 2 {
                return Err("quantized_conv2d requires at least 2 inputs".to_string());
            }
            let stride = get_int(attrs, "stride", 1) as usize;
            let padding = get_int(attrs, "padding", 0) as usize;
            let dilation = get_int(attrs, "dilation", 1) as usize;
            let groups = get_int(attrs, "groups", 1) as usize;
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            let result = crate::ops::conv_pool::conv2d::conv2d(
                &inputs[0], &inputs[1], bias, stride, padding, dilation, groups
            );
            Ok(vec![result])
        }
        "quantized_linear" => {
            if inputs.len() < 2 {
                return Err("quantized_linear requires at least 2 inputs".to_string());
            }
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            let w_t = crate::ops::linalg::transpose::transpose(&inputs[1]);
            let result = crate::ops::linalg::matmul::matmul(&inputs[0], &w_t);
            let result = if let Some(b) = bias {
                let mut data = result.data().to_vec();
                for (i, v) in data.iter_mut().enumerate() {
                    *v = *v + b.data()[i % b.len()];
                }
                Tensor::new(data, result.shape())
            } else {
                result
            };
            Ok(vec![result])
        }
        _ => Err(format!("Unknown quantized op: {}", op_type)),
    }
}

fn get_int(attrs: &HashMap<String, AttrValue>, key: &str, default: i32) -> i32 {
    attrs.get(key)
        .and_then(|v| match v {
            AttrValue::Int(i) => Some(*i as i32),
            AttrValue::IntList(list) if !list.is_empty() => Some(list[0] as i32),
            _ => None,
        })
        .unwrap_or(default)
}
