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

pub fn dispatch_index(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    match op_type {
        "topk" => {
            if inputs.is_empty() {
                return Err("topk requires 1 input".to_string());
            }
            let k = attrs
                .get("k")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as usize),
                    _ => None,
                })
                .unwrap_or(1);
            let dim = attrs
                .get("dim")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as usize),
                    _ => None,
                })
                .unwrap_or(0);
            let largest = attrs
                .get("largest")
                .and_then(|v| match v {
                    AttrValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(true);

            let (values, _indices) = crate::ops::embedding_lookup::topk::topk(
                &inputs[0], k, dim, largest,
            );
            Ok(vec![values])
        }
        _ => Err(format!("Unknown index op: {}", op_type)),
    }
}
