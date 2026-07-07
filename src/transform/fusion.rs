// src/transform/fusion.rs

use std::collections::HashMap;
use crate::ir::dag::{DagGraph, Op, AttrValue};

pub struct FusionPass;

impl FusionPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut new_ops = Vec::new();

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            if to_remove.contains(&op_id) {
                continue;
            }

            // 先取出 op 的数据，释放借用
            let op_data = match graph.ops.get(&op_id) {
                Some(op) => {
                    Some((
                        op_id,
                        op.op_type.clone(),
                        op.inputs.clone(),
                        op.outputs.clone(),
                        op.attrs.clone(),
                    ))
                }
                None => continue,
            };

            let (_, op_type, inputs, outputs, attrs) = match op_data {
                Some(data) => data,
                None => continue,
            };

            // 检查是否是 Conv2d
            if op_type != "conv2d" {
                continue;
            }

            // 用 inputs 而不是 op.inputs
            let conv_out = outputs[0];
            let users = graph.get_users(conv_out);
            if users.len() != 1 {
                continue;
            }

            let next_id = users[0];
            // 先获取 next_op 的数据
            let next_op_data = match graph.ops.get(&next_id) {
                Some(op) => {
                    Some((
                        next_id,
                        op.op_type.clone(),
                        op.inputs.clone(),
                        op.outputs.clone(),
                        op.attrs.clone(),
                    ))
                }
                None => continue,
            };

            let (next_id_clone, next_op_type, next_inputs, next_outputs, next_attrs) = match next_op_data {
                Some(data) => data,
                None => continue,
            };

            if next_op_type == "batchnorm2d" {
                // 传入克隆的数据
                if let Some(fused) = Self::fuse_conv_bn(
                    graph,
                    op_id,
                    &op_type, &inputs, &outputs, &attrs,
                    next_id_clone, &next_op_type, &next_inputs, &next_outputs, &next_attrs,
                    &mut to_remove
                ) {
                    new_ops.push(fused);
                    changed = true;
                }
            } else if next_op_type == "relu" {
                if let Some(fused) = Self::fuse_conv_relu(
                    graph,
                    op_id,
                    &op_type, &inputs, &outputs, &attrs,
                    next_id_clone, &next_op_type, &next_inputs, &next_outputs, &next_attrs,
                    &mut to_remove
                ) {
                    new_ops.push(fused);
                    changed = true;
                }
            }
        }

        // 删除被融合的算子
        for id in to_remove {
            graph.ops.remove(&id);
        }

        // 添加新算子
        for op in new_ops {
            graph.insert_op(op);
        }

        changed
    }

    fn fuse_conv_bn(
        graph: &mut DagGraph,
        conv_id: u64,
        _conv_type: &str,
        conv_inputs: &[u64],
        _conv_outputs: &[u64],
        conv_attrs: &HashMap<String, AttrValue>,
        bn_id: u64,
        _bn_type: &str,
        bn_inputs: &[u64],
        bn_outputs: &[u64],
        bn_attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let bn_out = bn_outputs[0];
        let users = graph.get_users(bn_out);
        let has_relu = users.len() == 1 &&
            graph.ops.get(&users[0]).map_or(false, |o| o.op_type == "relu");

        if bn_inputs.len() < 5 {
            return None;
        }

        // 修复未使用变量警告
        let _eps = bn_attrs.get("eps")
            .and_then(|v| match v { AttrValue::Float(f) => Some(*f), _ => None })
            .unwrap_or(1e-5);

        to_remove.push(conv_id);
        to_remove.push(bn_id);

        let fused_op = if has_relu {
            let relu_id = users[0];
            to_remove.push(relu_id);
            Self::create_fused_op(
                "fused_conv_bn_relu",
                conv_inputs.to_vec(),
                vec![bn_outputs[0]],
                conv_attrs.clone(),
                bn_attrs.clone(),
            )
        } else {
            Self::create_fused_op(
                "fused_conv_bn",
                conv_inputs.to_vec(),
                vec![bn_outputs[0]],
                conv_attrs.clone(),
                bn_attrs.clone(),
            )
        };

        Some(fused_op)
    }

    fn fuse_conv_relu(
        _graph: &mut DagGraph,
        conv_id: u64,
        _conv_type: &str,
        conv_inputs: &[u64],
        _conv_outputs: &[u64],
        conv_attrs: &HashMap<String, AttrValue>,
        relu_id: u64,
        _relu_type: &str,
        _relu_inputs: &[u64],
        relu_outputs: &[u64],
        relu_attrs: &HashMap<String, AttrValue>,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        to_remove.push(conv_id);
        to_remove.push(relu_id);

        let fused_op = Self::create_fused_op(
            "fused_conv_relu",
            conv_inputs.to_vec(),
            vec![relu_outputs[0]],
            conv_attrs.clone(),
            relu_attrs.clone(),
        );

        Some(fused_op)
    }

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
