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

pub fn dispatch_activation(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    match op_type {
        "relu" => {
            if inputs.is_empty() {
                return Err("relu requires 1 input".to_string());
            }
            let result = crate::ops::activation::relu::relu(&inputs[0]);
            Ok(vec![result])
        }
        "sigmoid" => {
            if inputs.is_empty() {
                return Err("sigmoid requires 1 input".to_string());
            }
            let result = crate::ops::activation::sigmoid::sigmoid(&inputs[0]);
            Ok(vec![result])
        }
        "tanh" => {
            if inputs.is_empty() {
                return Err("tanh requires 1 input".to_string());
            }
            let result = crate::ops::activation::tanh::tanh(&inputs[0]);
            Ok(vec![result])
        }
        "gelu" => {
            if inputs.is_empty() {
                return Err("gelu requires 1 input".to_string());
            }
            let result = crate::ops::activation::gelu::gelu(&inputs[0]);
            Ok(vec![result])
        }
        "silu" => {
            if inputs.is_empty() {
                return Err("silu requires 1 input".to_string());
            }
            let result = crate::ops::activation::silu::silu(&inputs[0]);
            Ok(vec![result])
        }
        "softmax" => {
            if inputs.is_empty() {
                return Err("softmax requires 1 input".to_string());
            }
            let dim = attrs
                .get("dim")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as usize),
                    _ => None,
                })
                .unwrap_or(0);
            let result =
                crate::ops::activation::softmax::softmax(&inputs[0], dim);
            Ok(vec![result])
        }
        "leaky_relu" => {
            if inputs.is_empty() {
                return Err("leaky_relu requires 1 input".to_string());
            }
            let alpha = attrs
                .get("alpha")
                .and_then(|v| match v {
                    AttrValue::Float(f) => Some(*f as f32),
                    _ => None,
                })
                .unwrap_or(0.01);
            let result = crate::ops::activation::leaky_relu::leaky_relu(
                &inputs[0], alpha,
            );
            Ok(vec![result])
        }
        "elu" => {
            if inputs.is_empty() {
                return Err("elu requires 1 input".to_string());
            }
            let alpha = attrs
                .get("alpha")
                .and_then(|v| match v {
                    AttrValue::Float(f) => Some(*f as f32),
                    _ => None,
                })
                .unwrap_or(1.0);
            let result = crate::ops::activation::elu::elu(&inputs[0], alpha);
            Ok(vec![result])
        }
        "relu6" => {
            if inputs.is_empty() {
                return Err("relu6 requires 1 input".to_string());
            }
            let result = crate::ops::activation::relu6::relu6(&inputs[0]);
            Ok(vec![result])
        }
        "log_softmax" => {
            if inputs.is_empty() {
                return Err("log_softmax requires 1 input".to_string());
            }
            let dim = attrs
                .get("dim")
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as usize),
                    _ => None,
                })
                .unwrap_or(0);
            let result = crate::ops::activation::log_softmax::log_softmax(
                &inputs[0], dim,
            );
            Ok(vec![result])
        }
        _ => Err(format!("Unknown activation op: {}", op_type)),
    }
}
