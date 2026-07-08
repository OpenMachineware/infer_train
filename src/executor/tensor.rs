use crate::ir::dag::AttrValue;
use crate::tensor::Tensor;
use std::collections::HashMap;

pub fn dispatch_tensor(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    match op_type {
        "reshape" => {
            if inputs.is_empty() {
                return Err("reshape requires 1 input".to_string());
            }
            let shape_attr = attrs.get("shape");
            if let Some(AttrValue::Shape(shape)) = shape_attr {
                let new_shape: Vec<usize> =
                    shape.iter().map(|&x| x as usize).collect();
                let result = crate::ops::tensor_manip::reshape::reshape(
                    &inputs[0], &new_shape,
                );
                Ok(vec![result])
            } else {
                Err("reshape requires shape attribute".to_string())
            }
        }
        "transpose" => {
            if inputs.is_empty() {
                return Err("transpose requires 1 input".to_string());
            }
            let result = crate::ops::linalg::transpose::transpose(&inputs[0]);
            Ok(vec![result])
        }
        "cat" => {
            if inputs.is_empty() {
                return Err("cat requires at least 1 input".to_string());
            }
            let dim = attrs
                .get("dim")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as usize),
                    _ => None,
                })
                .unwrap_or(0);
            // 收集引用
            let tensor_refs: Vec<&Tensor<f32>> = inputs.iter().collect();
            let result =
                crate::ops::tensor_manip::concat::concat(&tensor_refs, dim);
            Ok(vec![result])
        }
        _ => Err(format!("Unknown tensor op: {}", op_type)),
    }
}
