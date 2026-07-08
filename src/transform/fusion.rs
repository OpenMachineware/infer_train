use std::collections::HashMap;
use crate::ir::dag::{DagGraph, Op, DataType, TensorType, AttrValue};

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

        // 读取 BN 参数
        let eps = bn_attrs.get("eps")
            .and_then(|v| match v { AttrValue::Float(f) => Some(*f), _ => None })
            .unwrap_or(1e-5);

        // 获取 Conv 和 BN 的权重 ID
        let conv_weight_id = conv_inputs[1]; // conv 的第二个输入是 weight
        let conv_bias_id = if conv_inputs.len() >= 3 { Some(conv_inputs[2]) } else { None };
        let bn_weight_id = bn_inputs[1];
        let bn_bias_id = bn_inputs[2];
        let bn_mean_id = bn_inputs[3];
        let bn_var_id = bn_inputs[4];

        // 从 graph 中读取权重数据
        let conv_weight_data = graph.constants.get(&conv_weight_id)?.clone();
        let conv_bias_data = conv_bias_id
            .and_then(|id| graph.constants.get(&id))
            .map(|v| v.clone());
        let bn_weight_data = graph.constants.get(&bn_weight_id)?.clone();
        let bn_bias_data = graph.constants.get(&bn_bias_id)?.clone();
        let bn_mean_data = graph.constants.get(&bn_mean_id)?.clone();
        let bn_var_data = graph.constants.get(&bn_var_id)?.clone();

        // 获取 type 信息（在修改 graph 之前）
        let weight_ty = graph.values.get(&conv_weight_id).unwrap().ty.clone();
        let bias_ty = if let Some(bias_id) = conv_bias_id {
            graph.values.get(&bias_id).map(|v| v.ty.clone())
        } else {
            // 创建 bias type
            let out_channels = bn_weight_data.len() / 4;
            Some(TensorType {
                dtype: DataType::F32,
                shape: vec![out_channels as i64],
            })
        };

        // 执行权重合并
        let (fused_weight_data, fused_bias_data) = Self::fuse_conv_bn_weights(
            &conv_weight_data,
            conv_bias_data.as_ref().map(|v| v.as_slice()),
            &bn_weight_data,
            &bn_bias_data,
            &bn_mean_data,
            &bn_var_data,
            eps as f32,
        );

        // 创建融合后的权重常量
        let fused_weight_id = graph.add_constant(
            &format!("fused_conv_bn_weight_{}", conv_id),
            weight_ty,
            fused_weight_data,
        );

        let fused_bias_id = if let Some(bias_data) = fused_bias_data {
            if let Some(ty) = bias_ty {
                Some(graph.add_constant(
                    &format!("fused_conv_bn_bias_{}", conv_id),
                    ty,
                    bias_data,
                ))
            } else {
                None
            }
        } else {
            None
        };

        // 构建新的 inputs
        let mut new_inputs = vec![conv_inputs[0], fused_weight_id];
        if let Some(bias_id) = fused_bias_id {
            new_inputs.push(bias_id);
        }

        // 确定输出
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

    // Conv+BN 权重合并
    fn fuse_conv_bn_weights(
        conv_weight_data: &[u8],
        conv_bias_data: Option<&[u8]>,
        bn_weight_data: &[u8],
        bn_bias_data: &[u8],
        bn_mean_data: &[u8],
        bn_var_data: &[u8],
        eps: f32,
    ) -> (Vec<u8>, Option<Vec<u8>>) {
        // 解码为 f32
        let conv_weight: Vec<f32> = conv_weight_data.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let bn_weight: Vec<f32> = bn_weight_data.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bn_bias: Vec<f32> = bn_bias_data.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bn_mean: Vec<f32> = bn_mean_data.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bn_var: Vec<f32> = bn_var_data.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut fused_weight = conv_weight.clone();
        let mut fused_bias = if let Some(bias) = conv_bias_data {
            bias.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        } else {
            vec![0.0f32; bn_weight.len()]
        };

        // 合并公式:
        // weight_fused = weight_conv * bn_weight / sqrt(bn_var + eps)
        // bias_fused = (bias_conv - bn_mean) * bn_weight / sqrt(bn_var + eps) + bn_bias

        let out_channels = fused_bias.len();
        let elements_per_channel = conv_weight.len() / out_channels;

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

        // 编码回 u8
        let fused_weight_bytes: Vec<u8> = fused_weight.iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let fused_bias_bytes: Vec<u8> = fused_bias.iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();

        (fused_weight_bytes, Some(fused_bias_bytes))
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
