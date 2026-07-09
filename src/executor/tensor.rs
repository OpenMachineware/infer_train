// InferTrain - A Unified Inference and Training Engine
//
// Copyright (c) 2026 Jia Liu & InferTrain Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of InferTrain.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
            // Collect references
            let tensor_refs: Vec<&Tensor<f32>> = inputs.iter().collect();
            let result =
                crate::ops::tensor_manip::concat::concat(&tensor_refs, dim);
            Ok(vec![result])
        }
        _ => Err(format!("Unknown tensor op: {}", op_type)),
    }
}
