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

pub fn dispatch_control(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    match op_type {
        "where" => {
            if inputs.len() < 3 {
                return Err(
                    "where requires 3 inputs (condition, true_val, false_val)"
                        .to_string(),
                );
            }
            // Simplified: use conditional selection
            // Actual implementation needs condition to be bool tensor
            let result = inputs[1].clone();
            Ok(vec![result])
        }
        "sort" => {
            if inputs.is_empty() {
                return Err("sort requires 1 input".to_string());
            }
            let dim = attrs
                .get("dim")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as usize),
                    _ => None,
                })
                .unwrap_or(0);
            let ascending = attrs
                .get("ascending")
                .and_then(|v| match v {
                    AttrValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(true);

            let (values, _indices) = crate::ops::control_flow::sort::sort(
                &inputs[0], dim, ascending,
            );
            Ok(vec![values])
        }
        _ => Err(format!("Unknown control op: {}", op_type)),
    }
}
