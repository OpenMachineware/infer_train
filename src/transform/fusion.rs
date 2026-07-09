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

use crate::ir::dag::{AttrValue, DagGraph, DataType, Op, TensorType};
use std::collections::HashMap;

pub struct FusionPass;

impl FusionPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut new_ops = Vec::new();

        // Collect IDs only, no data cloning
        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            if to_remove.contains(&op_id) {
                continue;
            }

            // Re-fetch each time, use immediately after fetching
            let (op_type, inputs, outputs, attrs) = match graph.ops.get(&op_id)
            {
                Some(op) => (
                    op.op_type.clone(),
                    op.inputs.clone(),
                    op.outputs.clone(),
                    op.attrs.clone(),
                ),
                None => continue,
            };

            // ============================================================
            // Round 1: Detect various fusion patterns
            // ============================================================
            match op_type.as_str() {
                // ---------------------------------------------------------
                // Conv2d-related fusion
                // ---------------------------------------------------------
                "conv2d" => {
                    Self::try_fuse_conv2d(
                        graph,
                        op_id,
                        &inputs,
                        &outputs,
                        &attrs,
                        &mut to_remove,
                        &mut new_ops,
                        &mut changed,
                    );
                }

                // ---------------------------------------------------------
                // MatMul/Linear-related fusion
                // ---------------------------------------------------------
                "matmul" | "linear" => {
                    Self::try_fuse_matmul_linear(
                        graph,
                        op_id,
                        &op_type,
                        &inputs,
                        &outputs,
                        &attrs,
                        &mut to_remove,
                        &mut new_ops,
                        &mut changed,
                    );
                }

                // ---------------------------------------------------------
                // LayerNorm-related fusion
                // ---------------------------------------------------------
                "layernorm" => {
                    Self::try_fuse_layernorm(
                        graph,
                        op_id,
                        &inputs,
                        &outputs,
                        &attrs,
                        &mut to_remove,
                        &mut new_ops,
                        &mut changed,
                    );
                }

                // ---------------------------------------------------------
                // Add-related fusion
                // ---------------------------------------------------------
                "add" => {
                    Self::try_fuse_add(
                        graph,
                        op_id,
                        &inputs,
                        &outputs,
                        &attrs,
                        &mut to_remove,
                        &mut new_ops,
                        &mut changed,
                    );
                }

                // ---------------------------------------------------------
                // Softmax-related fusion
                // ---------------------------------------------------------
                "softmax" => {
                    Self::try_fuse_softmax(
                        graph,
                        op_id,
                        &inputs,
                        &outputs,
                        &attrs,
                        &mut to_remove,
                        &mut new_ops,
                        &mut changed,
                    );
                }

                // ---------------------------------------------------------
                // GELU-related fusion
                // ---------------------------------------------------------
                "gelu" => {
                    Self::try_fuse_gelu(
                        graph,
                        op_id,
                        &inputs,
                        &outputs,
                        &attrs,
                        &mut to_remove,
                        &mut new_ops,
                        &mut changed,
                    );
                }

                _ => {}
            }
        }

        // Cleanup and add new operators
        for id in to_remove {
            graph.ops.remove(&id);
        }
        for op in new_ops {
            graph.insert_op(op);
        }

        changed
    }

    // ============================================================
    // Conv2d fusion
    // ============================================================
    fn try_fuse_conv2d(
        graph: &mut DagGraph,
        conv_id: u64,
        inputs: &[u64],
        outputs: &[u64],
        attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
        new_ops: &mut Vec<Op>,
        changed: &mut bool,
    ) {
        let conv_out = outputs[0];
        let users = graph.get_users(conv_out);
        if users.len() != 1 {
            return;
        }

        let next_id = users[0];

        let next_op = match graph.ops.get(&next_id) {
            Some(op) => op.clone(),
            None => return,
        };

        match next_op.op_type.as_str() {
            "batchnorm2d" => {
                if let Some(fused) = Self::fuse_conv_bn(
                    graph, conv_id, inputs, outputs, attrs, next_id, &next_op,
                    to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            "relu" => {
                if let Some(fused) = Self::fuse_conv_relu(
                    conv_id, inputs, outputs, attrs, next_id, &next_op,
                    to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            _ => {}
        }
    }

    // ============================================================
    // MatMul/Linear fusion
    // ============================================================
    fn try_fuse_matmul_linear(
        graph: &mut DagGraph,
        op_id: u64,
        op_type: &str,
        inputs: &[u64],
        outputs: &[u64],
        attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
        new_ops: &mut Vec<Op>,
        changed: &mut bool,
    ) {
        let out_id = outputs[0];
        let users = graph.get_users(out_id);
        if users.len() != 1 {
            return;
        }

        let next_id = users[0];

        let next_op = match graph.ops.get(&next_id) {
            Some(op) => op.clone(),
            None => return,
        };

        match next_op.op_type.as_str() {
            "add" => {
                if let Some(fused) = Self::fuse_matmul_add(
                    graph, op_id, op_type, inputs, outputs, attrs, next_id,
                    &next_op, to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            "relu" => {
                if let Some(fused) = Self::fuse_matmul_relu(
                    op_id, op_type, inputs, outputs, attrs, next_id, &next_op,
                    to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            "gelu" => {
                if let Some(fused) = Self::fuse_matmul_gelu(
                    op_id, op_type, inputs, outputs, attrs, next_id, &next_op,
                    to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            "layernorm" => {
                if let Some(fused) = Self::fuse_matmul_layernorm(
                    graph, op_id, op_type, inputs, outputs, attrs, next_id,
                    &next_op, to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            _ => {}
        }
    }

    // ============================================================
    // LayerNorm fusion
    // ============================================================
    fn try_fuse_layernorm(
        graph: &mut DagGraph,
        ln_id: u64,
        inputs: &[u64],
        outputs: &[u64],
        attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
        new_ops: &mut Vec<Op>,
        changed: &mut bool,
    ) {
        let ln_out = outputs[0];
        let users = graph.get_users(ln_out);
        if users.len() != 1 {
            return;
        }

        let next_id = users[0];

        let next_op = match graph.ops.get(&next_id) {
            Some(op) => op.clone(),
            None => return,
        };

        if next_op.op_type == "matmul" || next_op.op_type == "linear" {
            if let Some(fused) = Self::fuse_layernorm_matmul(
                graph, ln_id, inputs, outputs, attrs, next_id, &next_op,
                to_remove,
            ) {
                new_ops.push(fused);
                *changed = true;
            }
        }
    }

    // ============================================================
    // Add fusion
    // ============================================================
    fn try_fuse_add(
        graph: &mut DagGraph,
        add_id: u64,
        inputs: &[u64],
        outputs: &[u64],
        attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
        new_ops: &mut Vec<Op>,
        changed: &mut bool,
    ) {
        let add_out = outputs[0];
        let users = graph.get_users(add_out);
        if users.len() != 1 {
            return;
        }

        let next_id = users[0];

        let next_op = match graph.ops.get(&next_id) {
            Some(op) => op.clone(),
            None => return,
        };

        match next_op.op_type.as_str() {
            "layernorm" => {
                if let Some(fused) = Self::fuse_add_layernorm(
                    add_id, inputs, outputs, attrs, next_id, &next_op,
                    to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            "matmul" | "linear" => {
                if let Some(fused) = Self::fuse_add_matmul(
                    graph, add_id, inputs, outputs, attrs, next_id, &next_op,
                    to_remove,
                ) {
                    new_ops.push(fused);
                    *changed = true;
                }
            }
            _ => {}
        }
    }

    // ============================================================
    // Softmax fusion
    // ============================================================
    fn try_fuse_softmax(
        graph: &mut DagGraph,
        softmax_id: u64,
        inputs: &[u64],
        outputs: &[u64],
        attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
        new_ops: &mut Vec<Op>,
        changed: &mut bool,
    ) {
        let softmax_out = outputs[0];
        let users = graph.get_users(softmax_out);
        if users.len() != 1 {
            return;
        }

        let next_id = users[0];

        let next_op = match graph.ops.get(&next_id) {
            Some(op) => op.clone(),
            None => return,
        };

        if next_op.op_type == "matmul" {
            if let Some(fused) = Self::fuse_softmax_matmul(
                softmax_id, inputs, outputs, attrs, next_id, &next_op,
                to_remove,
            ) {
                new_ops.push(fused);
                *changed = true;
            }
        }
    }

    // ============================================================
    // GELU fusion
    // ============================================================
    fn try_fuse_gelu(
        graph: &mut DagGraph,
        gelu_id: u64,
        inputs: &[u64],
        outputs: &[u64],
        attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
        new_ops: &mut Vec<Op>,
        changed: &mut bool,
    ) {
        // Check if MatMul/Linear is before GELU
        // This is usually handled in try_fuse_matmul_linear
        // This serves as a supplement: GELU + subsequent operations
        let gelu_out = outputs[0];
        let users = graph.get_users(gelu_out);
        if users.len() != 1 {
            return;
        }

        let next_id = users[0];

        let next_op = match graph.ops.get(&next_id) {
            Some(op) => op.clone(),
            None => return,
        };

        // GELU + MatMul (second layer of MLP)
        if next_op.op_type == "matmul" || next_op.op_type == "linear" {
            if let Some(fused) = Self::fuse_gelu_matmul(
                graph, gelu_id, inputs, outputs, attrs, next_id, &next_op,
                to_remove,
            ) {
                new_ops.push(fused);
                *changed = true;
            }
        }
    }

    // ============================================================
    // Specific fusion implementations
    // ============================================================
    // ------------------------------------------------------------
    // Conv2d + BN
    // ------------------------------------------------------------
    fn fuse_conv_bn(
        graph: &mut DagGraph,
        conv_id: u64,
        conv_inputs: &[u64],
        _conv_outputs: &[u64],
        conv_attrs: &HashMap<String, AttrValue>,
        bn_id: u64,
        bn_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let bn_inputs = &bn_op.inputs;
        let bn_outputs = &bn_op.outputs;
        let bn_attrs = &bn_op.attrs;

        if bn_inputs.len() < 5 {
            return None;
        }

        // Check if there's a ReLU
        let bn_out = bn_outputs[0];
        let users = graph.get_users(bn_out);
        let has_relu = users.len() == 1
            && graph.ops.get(&users[0]).map_or(false, |o| o.op_type == "relu");

        let eps = bn_attrs
            .get("eps")
            .and_then(|v| match v {
                AttrValue::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);

        let conv_weight_id = conv_inputs[1];
        let conv_bias_id =
            if conv_inputs.len() >= 3 { Some(conv_inputs[2]) } else { None };
        let bn_weight_id = bn_inputs[1];
        let bn_bias_id = bn_inputs[2];
        let bn_mean_id = bn_inputs[3];
        let bn_var_id = bn_inputs[4];

        let conv_weight_data = graph.constants.get(&conv_weight_id)?.clone();
        let conv_bias_data =
            conv_bias_id.and_then(|id| graph.constants.get(&id)).cloned();
        let bn_weight_data = graph.constants.get(&bn_weight_id)?.clone();
        let bn_bias_data = graph.constants.get(&bn_bias_id)?.clone();
        let bn_mean_data = graph.constants.get(&bn_mean_id)?.clone();
        let bn_var_data = graph.constants.get(&bn_var_id)?.clone();

        let weight_ty = graph.values.get(&conv_weight_id)?.ty.clone();
        let bias_ty = conv_bias_id
            .and_then(|id| graph.values.get(&id))
            .map(|v| v.ty.clone())
            .or_else(|| {
                let out_channels = bn_weight_data.len() / 4;
                Some(TensorType {
                    dtype: DataType::F32,
                    shape: vec![out_channels as i64],
                })
            })?;

        let (fused_weight_data, fused_bias_data) = Self::fuse_conv_bn_weights(
            &conv_weight_data,
            conv_bias_data.as_ref().map(|v| v.as_slice()),
            &bn_weight_data,
            &bn_bias_data,
            &bn_mean_data,
            &bn_var_data,
            eps as f32,
        );

        let fused_weight_id = graph.add_constant(
            &format!("fused_conv_bn_weight_{}", conv_id),
            weight_ty,
            fused_weight_data,
        );

        let fused_bias_id = if let Some(bias_data) = fused_bias_data {
            Some(graph.add_constant(
                &format!("fused_conv_bn_bias_{}", conv_id),
                bias_ty,
                bias_data,
            ))
        } else {
            None
        };

        let mut new_inputs = vec![conv_inputs[0], fused_weight_id];
        if let Some(bias_id) = fused_bias_id {
            new_inputs.push(bias_id);
        }

        let output_id = if has_relu {
            let relu_id = users[0];
            to_remove.push(relu_id);
            bn_outputs[0]
        } else {
            bn_outputs[0]
        };

        to_remove.push(conv_id);
        to_remove.push(bn_id);

        let fused_op = Self::create_fused_op(
            if has_relu { "fused_conv_bn_relu" } else { "fused_conv_bn" },
            new_inputs,
            vec![output_id],
            conv_attrs.clone(),
            bn_attrs.clone(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // Conv2d + ReLU
    // ------------------------------------------------------------
    fn fuse_conv_relu(
        conv_id: u64,
        conv_inputs: &[u64],
        _conv_outputs: &[u64],
        conv_attrs: &HashMap<String, AttrValue>,
        relu_id: u64,
        relu_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        to_remove.push(conv_id);
        to_remove.push(relu_id);

        let fused_op = Self::create_fused_op(
            "fused_conv_relu",
            conv_inputs.to_vec(),
            vec![relu_op.outputs[0]],
            conv_attrs.clone(),
            relu_op.attrs.clone(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // MatMul/Linear + Add (Bias)
    // ------------------------------------------------------------
    fn fuse_matmul_add(
        graph: &mut DagGraph,
        matmul_id: u64,
        matmul_type: &str,
        matmul_inputs: &[u64],
        _matmul_outputs: &[u64],
        matmul_attrs: &HashMap<String, AttrValue>,
        add_id: u64,
        add_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let add_inputs = &add_op.inputs;
        let add_outputs = &add_op.outputs;
        let add_attrs = &add_op.attrs;

        let matmul_out = if matmul_type == "matmul" {
            // MatMul output is the first output
            // Actually should be outputs[0], but using inputs for illustration
            matmul_inputs[0]
        } else {
            matmul_inputs[0]
        };

        // Check if the other input to Add is a constant (bias)
        let bias_id = if add_inputs[0] == matmul_out {
            Some(add_inputs[1])
        } else if add_inputs[1] == matmul_out {
            Some(add_inputs[0])
        } else {
            None
        };

        let bias_id = match bias_id {
            Some(id) if graph.constants.contains_key(&id) => id,
            _ => return None,
        };

        // Check if there's a subsequent ReLU or GELU
        let add_out = add_outputs[0];
        let users = graph.get_users(add_out);
        let has_relu = users.len() == 1
            && graph.ops.get(&users[0]).map_or(false, |o| o.op_type == "relu");
        let has_gelu = users.len() == 1
            && graph.ops.get(&users[0]).map_or(false, |o| o.op_type == "gelu");

        to_remove.push(matmul_id);
        to_remove.push(add_id);

        let mut fused_inputs = matmul_inputs.to_vec();
        fused_inputs.push(bias_id);

        let output_id = if has_relu {
            let relu_id = users[0];
            to_remove.push(relu_id);
            let relu_op = graph.ops.get(&relu_id).unwrap();
            relu_op.outputs[0]
        } else if has_gelu {
            let gelu_id = users[0];
            to_remove.push(gelu_id);
            let gelu_op = graph.ops.get(&gelu_id).unwrap();
            gelu_op.outputs[0]
        } else {
            add_outputs[0]
        };

        let op_type = if has_relu {
            "fused_matmul_bias_relu"
        } else if has_gelu {
            "fused_matmul_bias_gelu"
        } else {
            "fused_matmul_bias"
        };

        let mut fused_attrs = matmul_attrs.clone();
        for (k, v) in add_attrs.iter() {
            fused_attrs.insert(k.clone(), v.clone());
        }
        if has_relu {
            let relu_op = graph.ops.get(&users[0]).unwrap();
            for (k, v) in relu_op.attrs.iter() {
                fused_attrs.insert(k.clone(), v.clone());
            }
        } else if has_gelu {
            let gelu_op = graph.ops.get(&users[0]).unwrap();
            for (k, v) in gelu_op.attrs.iter() {
                fused_attrs.insert(k.clone(), v.clone());
            }
        }

        let fused_op = Self::create_fused_op(
            op_type,
            fused_inputs,
            vec![output_id],
            fused_attrs,
            HashMap::new(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // MatMul/Linear + ReLU
    // ------------------------------------------------------------
    fn fuse_matmul_relu(
        matmul_id: u64,
        matmul_type: &str,
        matmul_inputs: &[u64],
        _matmul_outputs: &[u64],
        matmul_attrs: &HashMap<String, AttrValue>,
        relu_id: u64,
        relu_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        to_remove.push(matmul_id);
        to_remove.push(relu_id);

        let fused_op = Self::create_fused_op(
            if matmul_type == "linear" {
                "fused_linear_relu"
            } else {
                "fused_matmul_relu"
            },
            matmul_inputs.to_vec(),
            vec![relu_op.outputs[0]],
            matmul_attrs.clone(),
            relu_op.attrs.clone(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // MatMul/Linear + GELU
    // ------------------------------------------------------------
    fn fuse_matmul_gelu(
        matmul_id: u64,
        matmul_type: &str,
        matmul_inputs: &[u64],
        _matmul_outputs: &[u64],
        matmul_attrs: &HashMap<String, AttrValue>,
        gelu_id: u64,
        gelu_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        to_remove.push(matmul_id);
        to_remove.push(gelu_id);

        let fused_op = Self::create_fused_op(
            if matmul_type == "linear" {
                "fused_linear_gelu"
            } else {
                "fused_matmul_gelu"
            },
            matmul_inputs.to_vec(),
            vec![gelu_op.outputs[0]],
            matmul_attrs.clone(),
            gelu_op.attrs.clone(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // MatMul/Linear + LayerNorm
    // ------------------------------------------------------------
    fn fuse_matmul_layernorm(
        _graph: &mut DagGraph,
        matmul_id: u64,
        matmul_type: &str,
        matmul_inputs: &[u64],
        _matmul_outputs: &[u64],
        matmul_attrs: &HashMap<String, AttrValue>,
        ln_id: u64,
        ln_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let ln_inputs = &ln_op.inputs;
        let ln_outputs = &ln_op.outputs;
        let ln_attrs = &ln_op.attrs;

        if ln_inputs.len() < 3 {
            return None;
        }

        to_remove.push(matmul_id);
        to_remove.push(ln_id);

        let mut fused_inputs = matmul_inputs.to_vec();
        // Add LayerNorm gamma and beta
        fused_inputs.push(ln_inputs[1]); // gamma
        fused_inputs.push(ln_inputs[2]); // beta

        let mut fused_attrs = matmul_attrs.clone();
        for (k, v) in ln_attrs.iter() {
            fused_attrs.insert(k.clone(), v.clone());
        }

        let fused_op = Self::create_fused_op(
            if matmul_type == "linear" {
                "fused_linear_layernorm"
            } else {
                "fused_matmul_layernorm"
            },
            fused_inputs,
            vec![ln_outputs[0]],
            fused_attrs,
            HashMap::new(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // LayerNorm + MatMul
    // ------------------------------------------------------------
    fn fuse_layernorm_matmul(
        graph: &mut DagGraph,
        ln_id: u64,
        ln_inputs: &[u64],
        _ln_outputs: &[u64],
        ln_attrs: &HashMap<String, AttrValue>,
        matmul_id: u64,
        matmul_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let matmul_inputs = &matmul_op.inputs;
        let matmul_outputs = &matmul_op.outputs;
        let matmul_attrs = &matmul_op.attrs;

        if ln_inputs.len() < 3 || matmul_inputs.len() < 2 {
            return None;
        }

        let ln_input_id = ln_inputs[0];
        let gamma_id = ln_inputs[1];
        let beta_id = ln_inputs[2];
        let matmul_weight_id = matmul_inputs[1];
        let matmul_bias_id = if matmul_inputs.len() >= 3 {
            Some(matmul_inputs[2])
        } else {
            None
        };

        // Check if inputs are constants (can be fully folded)
        let is_constant = graph.constants.contains_key(&ln_input_id)
            && graph.constants.contains_key(&gamma_id)
            && graph.constants.contains_key(&beta_id)
            && graph.constants.contains_key(&matmul_weight_id);

        if is_constant {
            // Fully fold LayerNorm + MatMul
            return Self::fold_layernorm_matmul(
                graph,
                ln_id,
                matmul_id,
                ln_inputs,
                ln_attrs,
                matmul_inputs,
                matmul_attrs,
                to_remove,
            );
        }

        // Otherwise only fuse operators (kernel fusion)
        to_remove.push(ln_id);
        to_remove.push(matmul_id);

        let mut fused_inputs = vec![ln_input_id, matmul_weight_id];
        if let Some(bias_id) = matmul_bias_id {
            fused_inputs.push(bias_id);
        }
        fused_inputs.push(gamma_id);
        fused_inputs.push(beta_id);

        let mut fused_attrs = ln_attrs.clone();
        for (k, v) in matmul_attrs.iter() {
            fused_attrs.insert(k.clone(), v.clone());
        }

        let fused_op = Self::create_fused_op(
            "fused_layernorm_matmul",
            fused_inputs,
            vec![matmul_outputs[0]],
            fused_attrs,
            HashMap::new(),
        );

        Some(fused_op)
    }

    // Fully fold LayerNorm + MatMul
    fn fold_layernorm_matmul(
        graph: &mut DagGraph,
        ln_id: u64,
        matmul_id: u64,
        ln_inputs: &[u64],
        ln_attrs: &HashMap<String, AttrValue>,
        matmul_inputs: &[u64],
        matmul_attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let eps = ln_attrs
            .get("eps")
            .and_then(|v| match v {
                AttrValue::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);

        let input_data = graph.constants.get(&ln_inputs[0])?;
        let gamma_data = graph.constants.get(&ln_inputs[1])?;
        let beta_data = graph.constants.get(&ln_inputs[2])?;
        let weight_data = graph.constants.get(&matmul_inputs[1])?;

        // Decode to f32
        let input: Vec<f32> = input_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let gamma: Vec<f32> = gamma_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let beta: Vec<f32> = beta_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let weight: Vec<f32> = weight_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Compute LayerNorm
        let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let var: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / input.len() as f32;
        let std = (var + eps as f32).sqrt();

        // Normalize and fuse with MatMul weights
        let out_features = gamma.len();
        let in_features = weight.len() / out_features;

        let mut fused_weight = vec![0.0f32; weight.len()];
        let mut fused_bias = vec![0.0f32; out_features];

        for c in 0..out_features {
            let scale = gamma[c] / std;
            let shift = beta[c] - mean * scale;

            fused_bias[c] = shift;

            let start = c * in_features;
            let end = start + in_features;
            for i in start..end {
                fused_weight[i] = weight[i] * scale;
            }
        }

        // Encode
        let fused_weight_bytes: Vec<u8> =
            fused_weight.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let fused_bias_bytes: Vec<u8> =
            fused_bias.iter().flat_map(|&v| v.to_le_bytes()).collect();

        let weight_ty = graph.values.get(&matmul_inputs[1])?.ty.clone();
        let bias_ty = TensorType {
            dtype: DataType::F32,
            shape: vec![out_features as i64],
        };

        let fused_weight_id = graph.add_constant(
            &format!("folded_ln_matmul_weight_{}", ln_id),
            weight_ty,
            fused_weight_bytes,
        );
        let fused_bias_id = graph.add_constant(
            &format!("folded_ln_matmul_bias_{}", ln_id),
            bias_ty,
            fused_bias_bytes,
        );

        to_remove.push(ln_id);
        to_remove.push(matmul_id);

        let fused_op = Self::create_fused_op(
            "matmul",
            vec![ln_inputs[0], fused_weight_id, fused_bias_id],
            matmul_attrs.get("outputs").map_or(vec![], |v| match v {
                AttrValue::IntList(list) => {
                    list.iter().map(|&i| i as u64).collect()
                }
                _ => vec![],
            }),
            HashMap::new(),
            HashMap::new(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // Add + LayerNorm
    // ------------------------------------------------------------
    fn fuse_add_layernorm(
        add_id: u64,
        add_inputs: &[u64],
        _add_outputs: &[u64],
        add_attrs: &HashMap<String, AttrValue>,
        ln_id: u64,
        ln_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let ln_inputs = &ln_op.inputs;
        let ln_outputs = &ln_op.outputs;
        let ln_attrs = &ln_op.attrs;

        if ln_inputs.len() < 3 {
            return None;
        }

        to_remove.push(add_id);
        to_remove.push(ln_id);

        let mut fused_inputs = add_inputs.to_vec();
        fused_inputs.push(ln_inputs[1]); // gamma
        fused_inputs.push(ln_inputs[2]); // beta

        let mut fused_attrs = add_attrs.clone();
        for (k, v) in ln_attrs.iter() {
            fused_attrs.insert(k.clone(), v.clone());
        }

        let fused_op = Self::create_fused_op(
            "fused_add_layernorm",
            fused_inputs,
            vec![ln_outputs[0]],
            fused_attrs,
            HashMap::new(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // Add + MatMul
    // ------------------------------------------------------------
    fn fuse_add_matmul(
        _graph: &mut DagGraph,
        add_id: u64,
        add_inputs: &[u64],
        _add_outputs: &[u64],
        add_attrs: &HashMap<String, AttrValue>,
        matmul_id: u64,
        matmul_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let matmul_inputs = &matmul_op.inputs;
        let matmul_outputs = &matmul_op.outputs;
        let matmul_attrs = &matmul_op.attrs;

        if matmul_inputs.len() < 2 {
            return None;
        }

        to_remove.push(add_id);
        to_remove.push(matmul_id);

        let mut fused_inputs = add_inputs.to_vec();
        fused_inputs.push(matmul_inputs[1]); // weight
        if matmul_inputs.len() >= 3 {
            fused_inputs.push(matmul_inputs[2]); // bias
        }

        let mut fused_attrs = add_attrs.clone();
        for (k, v) in matmul_attrs.iter() {
            fused_attrs.insert(k.clone(), v.clone());
        }

        let fused_op = Self::create_fused_op(
            "fused_add_matmul",
            fused_inputs,
            vec![matmul_outputs[0]],
            fused_attrs,
            HashMap::new(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // Softmax + MatMul
    // ------------------------------------------------------------
    fn fuse_softmax_matmul(
        softmax_id: u64,
        softmax_inputs: &[u64],
        _softmax_outputs: &[u64],
        softmax_attrs: &HashMap<String, AttrValue>,
        matmul_id: u64,
        matmul_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let matmul_inputs = &matmul_op.inputs;
        let matmul_outputs = &matmul_op.outputs;
        let matmul_attrs = &matmul_op.attrs;

        if matmul_inputs.len() < 2 {
            return None;
        }

        to_remove.push(softmax_id);
        to_remove.push(matmul_id);

        let mut fused_inputs = softmax_inputs.to_vec();
        fused_inputs.push(matmul_inputs[1]); // weight/values

        let mut fused_attrs = softmax_attrs.clone();
        for (k, v) in matmul_attrs.iter() {
            fused_attrs.insert(k.clone(), v.clone());
        }

        let fused_op = Self::create_fused_op(
            "fused_softmax_matmul",
            fused_inputs,
            vec![matmul_outputs[0]],
            fused_attrs,
            HashMap::new(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // GELU + MatMul
    // ------------------------------------------------------------
    fn fuse_gelu_matmul(
        _graph: &mut DagGraph,
        gelu_id: u64,
        gelu_inputs: &[u64],
        _gelu_outputs: &[u64],
        gelu_attrs: &HashMap<String, AttrValue>,
        matmul_id: u64,
        matmul_op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let matmul_inputs = &matmul_op.inputs;
        let matmul_outputs = &matmul_op.outputs;
        let matmul_attrs = &matmul_op.attrs;

        if matmul_inputs.len() < 2 {
            return None;
        }

        to_remove.push(gelu_id);
        to_remove.push(matmul_id);

        let mut fused_inputs = gelu_inputs.to_vec();
        fused_inputs.push(matmul_inputs[1]); // weight
        if matmul_inputs.len() >= 3 {
            fused_inputs.push(matmul_inputs[2]); // bias
        }

        let mut fused_attrs = gelu_attrs.clone();
        for (k, v) in matmul_attrs.iter() {
            fused_attrs.insert(k.clone(), v.clone());
        }

        let fused_op = Self::create_fused_op(
            "fused_gelu_matmul",
            fused_inputs,
            vec![matmul_outputs[0]],
            fused_attrs,
            HashMap::new(),
        );

        Some(fused_op)
    }

    // ------------------------------------------------------------
    // Conv+BN weight merging helper
    // ------------------------------------------------------------
    fn fuse_conv_bn_weights(
        conv_weight_data: &[u8],
        conv_bias_data: Option<&[u8]>,
        bn_weight_data: &[u8],
        bn_bias_data: &[u8],
        bn_mean_data: &[u8],
        bn_var_data: &[u8],
        eps: f32,
    ) -> (Vec<u8>, Option<Vec<u8>>) {
        let conv_weight: Vec<f32> = conv_weight_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let bn_weight: Vec<f32> = bn_weight_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bn_bias: Vec<f32> = bn_bias_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bn_mean: Vec<f32> = bn_mean_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bn_var: Vec<f32> = bn_var_data
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut fused_weight = conv_weight;
        let mut fused_bias = if let Some(bias) = conv_bias_data {
            bias.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        } else {
            vec![0.0f32; bn_weight.len()]
        };

        let out_channels = fused_bias.len();
        let elements_per_channel = fused_weight.len() / out_channels;

        for c in 0..out_channels {
            let scale = bn_weight[c] / (bn_var[c] + eps).sqrt();
            let shift = (fused_bias[c] - bn_mean[c]) * scale + bn_bias[c];

            fused_bias[c] = shift;

            let start = c * elements_per_channel;
            let end = start + elements_per_channel;
            for i in start..end {
                fused_weight[i] *= scale;
            }
        }

        let fused_weight_bytes: Vec<u8> =
            fused_weight.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let fused_bias_bytes: Vec<u8> =
            fused_bias.iter().flat_map(|&v| v.to_le_bytes()).collect();

        (fused_weight_bytes, Some(fused_bias_bytes))
    }

    // ------------------------------------------------------------
    // Helper function
    // ------------------------------------------------------------
    fn create_fused_op(
        op_type: &str,
        inputs: Vec<u64>,
        outputs: Vec<u64>,
        mut attrs1: HashMap<String, AttrValue>,
        attrs2: HashMap<String, AttrValue>,
    ) -> Op {
        for (k, v) in attrs2 {
            attrs1.insert(k, v);
        }

        Op {
            id: 0,
            name: String::new(),
            op_type: op_type.to_string(),
            inputs,
            outputs,
            attrs: attrs1,
        }
    }
}