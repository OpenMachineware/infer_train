// src/transform/cse.rs

use crate::ir::dag::{AttrValue, DagGraph, Op};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub struct CommonSubexpressionEliminationPass;

impl CommonSubexpressionEliminationPass {
    pub fn apply(graph: &mut DagGraph) -> bool {
        let mut changed = false;
        // let mut seen: HashMap<u64, String> = HashMap::new();  // op_id -> hash
        let mut hash_to_op: HashMap<String, u64> = HashMap::new(); // hash -> op_id
        let mut to_remove = Vec::new();
        let mut replacements: HashMap<u64, u64> = HashMap::new();

        // 按拓扑顺序遍历
        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o,
                None => continue,
            };

            // 只处理纯函数算子（没有副作用的）
            if Self::is_pure_operator(&op.op_type) {
                let hash = Self::hash_op(op);

                if let Some(&existing_id) = hash_to_op.get(&hash) {
                    // 找到重复的算子
                    // 将当前算子的输出替换为已有算子的输出
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

        // 应用替换
        for (old_id, new_id) in &replacements {
            // 更新所有算子的 inputs
            for (_, op) in graph.ops.iter_mut() {
                for input in &mut op.inputs {
                    if input == old_id {
                        *input = *new_id;
                    }
                }
            }
            // 更新 graph.outputs
            for output in &mut graph.outputs {
                if output == old_id {
                    *output = *new_id;
                }
            }
        }

        // 删除被替换的算子
        for id in to_remove {
            graph.ops.remove(&id);
        }

        // 清理没有被任何算子使用的 Value（除了 inputs 和 outputs）
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
        // 纯函数算子：没有副作用，相同输入产生相同输出
        match op_type {
            // 数学算子
            "add" | "sub" | "mul" | "div" | "pow" => true,
            "exp" | "sqrt" | "log" | "log2" | "log10" => true,
            "abs" | "neg" | "clamp" | "floor" | "ceil" | "round" => true,
            // 激活函数
            "relu" | "leaky_relu" | "elu" | "gelu" | "relu6" => true,
            "sigmoid" | "tanh" | "silu" | "hard_swish" | "hard_sigmoid" => true,
            "softplus" | "softshrink" | "celu" => true,
            "softmax" | "log_softmax" => true,
            // 张量操作
            "reshape" | "transpose" | "slice" => true,
            "cumsum" | "cumprod" => true,
            // 线性层
            "linear" => true,
            // 池化
            "maxpool2d" | "avgpool2d" => true,
            // 归一化（推理模式是纯函数）
            "batchnorm2d" | "layernorm" | "rmsnorm" => true,
            // 其他
            "matmul" => true,
            "cat" => true,
            // 有副作用的（训练相关）
            "dropout" => false, // 训练时 dropout 有随机性
            _ => false,
        }
    }

    fn hash_op(op: &Op) -> String {
        // 计算算子的哈希：op_type + inputs + attrs
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
                // float 转 bits 后 hash
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
