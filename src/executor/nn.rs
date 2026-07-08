// src/executor/nn.rs

use crate::ir::dag::AttrValue;
use crate::tensor::Tensor;
use std::collections::HashMap;

pub fn dispatch_nn(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    match op_type {
        "conv2d" => {
            if inputs.len() < 2 {
                return Err("conv2d requires at least 2 inputs".to_string());
            }
            let stride = get_int(attrs, "stride", 1) as usize;
            let padding = get_int(attrs, "padding", 0) as usize;
            let dilation = get_int(attrs, "dilation", 1) as usize;
            let groups = get_int(attrs, "groups", 1) as usize;
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            let result = crate::ops::conv_pool::conv2d::conv2d(
                &inputs[0], &inputs[1], bias, stride, padding, dilation, groups,
            );
            Ok(vec![result])
        }
        "linear" => {
            if inputs.len() < 2 {
                return Err("linear requires at least 2 inputs".to_string());
            }
            let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
            // linear = matmul(x, weight.T) + bias
            let w_t = crate::ops::linalg::transpose::transpose(&inputs[1]);
            let result = crate::ops::linalg::matmul::matmul(&inputs[0], &w_t);
            let result = if let Some(b) = bias {
                // 加上 bias
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
        "maxpool2d" => {
            if inputs.is_empty() {
                return Err("maxpool2d requires 1 input".to_string());
            }
            let kernel_size = get_int(attrs, "kernel_size", 2) as usize;
            let stride = get_int(attrs, "stride", kernel_size as i32) as usize;
            let padding = get_int(attrs, "padding", 0) as usize;
            let result = crate::ops::conv_pool::max_pool::max_pool(
                &inputs[0],
                kernel_size,
                stride,
                padding,
            );
            Ok(vec![result])
        }
        "avgpool2d" => {
            if inputs.is_empty() {
                return Err("avgpool2d requires 1 input".to_string());
            }
            let kernel_size = get_int(attrs, "kernel_size", 2) as usize;
            let stride = get_int(attrs, "stride", kernel_size as i32) as usize;
            let padding = get_int(attrs, "padding", 0) as usize;
            let result = crate::ops::conv_pool::avg_pool::avg_pool(
                &inputs[0],
                kernel_size,
                stride,
                padding,
            );
            Ok(vec![result])
        }
        "batchnorm2d" => {
            if inputs.len() < 5 {
                return Err("batchnorm2d requires 5 inputs".to_string());
            }
            let eps = get_float(attrs, "eps", 1e-5);
            let result = crate::ops::normalization::batch_norm::batch_norm(
                &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], eps,
            );
            Ok(vec![result])
        }
        "layernorm" => {
            if inputs.len() < 3 {
                return Err("layernorm requires 3 inputs".to_string());
            }
            let eps = get_float(attrs, "eps", 1e-5);
            let result = crate::ops::normalization::layer_norm::layer_norm(
                &inputs[0], &inputs[1], &inputs[2], eps,
            );
            Ok(vec![result])
        }
        "dropout" => {
            if inputs.is_empty() {
                return Err("dropout requires 1 input".to_string());
            }
            let _p = get_float(attrs, "p", 0.5);
            let training = get_bool(attrs, "training", false);
            // 推理模式下 dropout 是 identity
            if training {
                // 简化版：返回原值
                Ok(vec![inputs[0].clone()])
            } else {
                Ok(vec![inputs[0].clone()])
            }
        }
        "embedding" => {
            if inputs.len() < 2 {
                return Err(
                    "embedding requires 2 inputs (indices, weight)".to_string()
                );
            }
            // 需要从 indices 提取 i64
            // 暂时简化
            Ok(vec![inputs[1].clone()])
        }
        _ => Err(format!("Unknown nn op: {}", op_type)),
    }
}

fn get_int(attrs: &HashMap<String, AttrValue>, key: &str, default: i32) -> i32 {
    attrs
        .get(key)
        .and_then(|v| match v {
            AttrValue::Int(i) => Some(*i as i32),
            AttrValue::IntList(list) if !list.is_empty() => {
                Some(list[0] as i32)
            }
            _ => None,
        })
        .unwrap_or(default)
}

fn get_float(
    attrs: &HashMap<String, AttrValue>,
    key: &str,
    default: f32,
) -> f32 {
    attrs
        .get(key)
        .and_then(|v| match v {
            AttrValue::Float(f) => Some(*f as f32),
            AttrValue::Int(i) => Some(*i as f32),
            _ => None,
        })
        .unwrap_or(default)
}

fn get_bool(
    attrs: &HashMap<String, AttrValue>,
    key: &str,
    default: bool,
) -> bool {
    attrs
        .get(key)
        .and_then(|v| match v {
            AttrValue::Bool(b) => Some(*b),
            AttrValue::Int(i) => Some(*i != 0),
            _ => None,
        })
        .unwrap_or(default)
}
