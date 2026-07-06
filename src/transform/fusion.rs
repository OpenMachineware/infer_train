// src/transform/fusion.rs

use std::collections::HashMap;
use crate::ir::dag::{DagGraph, Op, AttrValue};

pub struct FusionPass;

impl FusionPass {
    pub fn apply(&self, graph: &mut DagGraph) -> bool {
        let mut changed = false;
        let mut to_remove = Vec::new();
        let mut new_ops = Vec::new();

        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            if to_remove.contains(&op_id) {
                continue;
            }

            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            if let Some(fused) = self.try_fuse(graph, op_id, op, &mut to_remove) {
                new_ops.push(fused);
                changed = true;
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

    // ============================================================
    // 融合模式：Conv2d + BatchNorm + ReLU
    // ============================================================
    fn try_fuse(
        &self,
        graph: &mut DagGraph,
        op_id: u64,
        op: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        if op.op_type != "conv2d" {
            return None;
        }

        let conv_out = op.outputs[0];
        let users = graph.get_users(conv_out);
        if users.len() != 1 {
            return None;
        }

        let next_id = users[0];
        let next_op = match graph.ops.get(&next_id) {
            Some(o) => o,
            None => return None,
        };

        if next_op.op_type == "batchnorm2d" {
            return self.fuse_conv_bn(graph, op_id, op, next_id, next_op, to_remove);
        }

        if next_op.op_type == "relu" {
            return self.fuse_conv_relu(graph, op_id, op, next_id, next_op, to_remove);
        }

        None
    }

    // ============================================================
    // Conv2d + BatchNorm 融合
    // ============================================================
    fn fuse_conv_bn(
        &self,
        graph: &mut DagGraph,
        conv_id: u64,
        conv: &Op,
        bn_id: u64,
        bn: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        let bn_out = bn.outputs[0];
        let users = graph.get_users(bn_out);
        let has_relu = users.len() == 1 &&
            graph.ops.get(&users[0]).map_or(false, |o| o.op_type == "relu");

        if bn.inputs.len() < 5 {
            return None;
        }

        let eps = bn.attrs.get("eps")
            .and_then(|v| match v { AttrValue::Float(f) => Some(*f), _ => None })
            .unwrap_or(1e-5);

        to_remove.push(conv_id);
        to_remove.push(bn_id);

        let fused_op = if has_relu {
            let relu_id = users[0];
            to_remove.push(relu_id);
            self.create_fused_op(
                graph,
                "fused_conv_bn_relu",
                conv.inputs.clone(),
                vec![bn.outputs[0]],
                conv.attrs.clone(),
                bn.attrs.clone(),
            )
        } else {
            self.create_fused_op(
                graph,
                "fused_conv_bn",
                conv.inputs.clone(),
                vec![bn.outputs[0]],
                conv.attrs.clone(),
                bn.attrs.clone(),
            )
        };

        Some(fused_op)
    }

    // ============================================================
    // Conv2d + ReLU 融合
    // ============================================================
    fn fuse_conv_relu(
        &self,
        graph: &mut DagGraph,
        conv_id: u64,
        conv: &Op,
        relu_id: u64,
        relu: &Op,
        to_remove: &mut Vec<u64>,
    ) -> Option<Op> {
        to_remove.push(conv_id);
        to_remove.push(relu_id);

        let fused_op = self.create_fused_op(
            graph,
            "fused_conv_relu",
            conv.inputs.clone(),
            vec![relu.outputs[0]],
            conv.attrs.clone(),
            relu.attrs.clone(),
        );

        Some(fused_op)
    }

    // ============================================================
    // 工具函数
    // ============================================================
    fn create_fused_op(
        &self,
        graph: &mut DagGraph,
        op_type: &str,
        inputs: Vec<u64>,
        outputs: Vec<u64>,
        mut attrs1: HashMap<String, AttrValue>,
        attrs2: HashMap<String, AttrValue>,
    ) -> Op {
        for (k, v) in attrs2 {
            attrs1.insert(k, v);
        }

        // 让 graph.insert_op 自动分配ID
        Op {
            id: 0, // 临时值，insert_op会重新分配
            name: String::new(), // 临时值，insert_op会设置
            op_type: op_type.to_string(),
            inputs,
            outputs,
            attrs: attrs1,
        }
    }
}
