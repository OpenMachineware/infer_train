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

use crate::ir::dag::{AttrValue, DagGraph, Op};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub struct CommonSubexpressionEliminationPass;

impl CommonSubexpressionEliminationPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        // op_id -> hash
        // let mut seen: HashMap<u64, String> = HashMap::new();
        // hash -> op_id
        let mut hash_to_op: HashMap<String, u64> = HashMap::new();
        let mut to_remove = Vec::new();
        let mut replacements: HashMap<u64, u64> = HashMap::new();

        // Traverse in topological order
        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            // Only process pure function operators (no side effects)
            if Self::is_pure_operator(&op.op_type) {
                let hash = Self::hash_op(op);

                if let Some(&existing_id) = hash_to_op.get(&hash) {
                    // Found duplicate operator
                    // Replace current operator's output with existing
                    // operator's output
                    if op.outputs.len() == 1 && existing_id != op_id {
                        let old_out = op.outputs[0];
                        let new_out =
                            graph.ops.get(&existing_id).unwrap().outputs[0];
                        replacements.insert(old_out, new_out);
                        to_remove.push(op_id);
                        changed = true;
                    }
                } else {
                    hash_to_op.insert(hash.clone(), op_id);
                    // seen.insert(op_id, hash);
                }
            }
        }

        // Apply replacements
        for (old_id, new_id) in &replacements {
            // Update all operators' inputs
            for (_, op) in graph.ops.iter_mut() {
                for input in &mut op.inputs {
                    if input == old_id {
                        *input = *new_id;
                    }
                }
            }
            // Update graph.outputs
            for output in &mut graph.outputs {
                if output == old_id {
                    *output = *new_id;
                }
            }
        }

        // Remove replaced operators
        for id in to_remove {
            graph.ops.remove(&id);
        }

        // Clean up Values not used by any operator (except inputs and outputs)
        let live_values: Vec<u64> =
            graph.inputs.iter().chain(graph.outputs.iter()).cloned().collect();
        let used_values: Vec<u64> = graph
            .ops
            .values()
            .flat_map(|op| op.inputs.iter().cloned())
            .chain(graph.ops.values().flat_map(|op| op.outputs.iter().cloned()))
            .collect();

        let dead_values: Vec<u64> = graph
            .values
            .keys()
            .filter(|&id| {
                !live_values.contains(id) && !used_values.contains(id)
            })
            .cloned()
            .collect();

        for id in dead_values {
            graph.values.remove(&id);
            graph.constants.remove(&id);
        }

        changed
    }

    fn is_pure_operator(op_type: &str) -> bool {
        // Pure function operators: no side effects,
        // same input produces same output
        match op_type {
            // Math operators
            "add" | "sub" | "mul" | "div" | "pow" => true,
            "exp" | "sqrt" | "log" | "log2" | "log10" => true,
            "abs" | "neg" | "clamp" | "floor" | "ceil" | "round" => true,
            // Activation functions
            "relu" | "leaky_relu" | "elu" | "gelu" | "relu6" => true,
            "sigmoid" | "tanh" | "silu" | "hard_swish" | "hard_sigmoid" => true,
            "softplus" | "softshrink" | "celu" => true,
            "softmax" | "log_softmax" => true,
            // Tensor operations
            "reshape" | "transpose" | "slice" => true,
            "cumsum" | "cumprod" => true,
            // Linear layers
            "linear" => true,
            // Pooling
            "maxpool2d" | "avgpool2d" => true,
            // Normalization (pure function in inference mode)
            "batchnorm2d" | "layernorm" | "rmsnorm" => true,
            // Others
            "matmul" => true,
            "cat" => true,
            // Side-effect operators (training related)
            "dropout" => false, // dropout has randomness during training
            _ => false,
        }
    }

    fn hash_op(op: &Op) -> String {
        // Compute operator hash: op_type + inputs + attrs
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        op.op_type.hash(&mut hasher);
        op.inputs.hash(&mut hasher);
        for (key, value) in &op.attrs {
            key.hash(&mut hasher);
            Self::hash_attr(value, &mut hasher);
        }
        format!("{:016x}", hasher.finish())
    }

    fn hash_attr(
        attr: &AttrValue,
        hasher: &mut std::collections::hash_map::DefaultHasher,
    ) {
        match attr {
            AttrValue::Int(i) => i.hash(hasher),
            AttrValue::Float(f) => {
                // hash float after converting to bits
                f.to_bits().hash(hasher)
            }
            AttrValue::Bool(b) => b.hash(hasher),
            AttrValue::String(s) => s.hash(hasher),
            AttrValue::IntList(list) => list.hash(hasher),
            AttrValue::FloatList(list) => {
                for f in list {
                    f.to_bits().hash(hasher);
                }
            }
            AttrValue::Shape(shape) => shape.hash(hasher),
        }
    }
}
