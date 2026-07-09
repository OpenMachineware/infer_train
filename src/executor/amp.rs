// src/executor/amp.rs

use crate::ir::dag::{AttrValue, DagGraph, DataType, Op, TensorType};
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmpDtype {
    F16,
    BF16,
}

impl AmpDtype {
    pub fn to_data_type(&self) -> DataType {
        match self {
            AmpDtype::F16 => DataType::F16,
            AmpDtype::BF16 => DataType::BF16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmpConfig {
    pub enabled: bool,
    pub dtype: AmpDtype,
    pub loss_scaling: bool,
    pub init_scale: f32,
    pub dynamic_scale: bool,
}

impl Default for AmpConfig {
    fn default() -> Self {
        AmpConfig {
            enabled: false,
            dtype: AmpDtype::F16,
            loss_scaling: true,
            init_scale: 1024.0,
            dynamic_scale: true,
        }
    }
}

impl AmpConfig {
    pub fn inference() -> Self {
        AmpConfig {
            enabled: false,
            dtype: AmpDtype::F16,
            loss_scaling: false,
            init_scale: 1.0,
            dynamic_scale: false,
        }
    }

    pub fn training() -> Self {
        AmpConfig {
            enabled: true,
            dtype: AmpDtype::F16,
            loss_scaling: true,
            init_scale: 1024.0,
            dynamic_scale: true,
        }
    }

    pub fn bf16() -> Self {
        AmpConfig {
            enabled: true,
            dtype: AmpDtype::BF16,
            loss_scaling: false,
            init_scale: 1.0,
            dynamic_scale: false,
        }
    }
}

// ============================================================
// AMP 图转换器
// ============================================================

pub struct AmpGraphConverter;

impl AmpGraphConverter {
    /// 将 DAG 转换为混合精度 DAG
    /// 规则：
    /// 1. 输入插入 Cast (FP32 → AMP dtype)
    /// 2. 输出插入 Cast (AMP dtype → FP32)
    /// 3. 每个算子的输入/输出自动适配 AMP dtype
    pub fn convert(
        graph: &mut DagGraph,
        config: &AmpConfig,
        _param_ids: &[u64],
    ) -> Result<(), String> {
        if !config.enabled {
            return Ok(());
        }

        let amp_dtype = config.dtype.to_data_type();
        let mut new_ops = Vec::new();
        let mut to_remove = Vec::new();
        let mut next_id = graph.next_id;

        // 收集所有需要转换的节点
        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o.clone(),
                None => continue,
            };

            // 判断算子是否应该在 AMP dtype 下执行
            if Self::should_run_in_amp(&op.op_type) {
                // 为每个输入插入 Cast (FP32 → AMP dtype)
                let mut new_inputs = Vec::new();
                for &in_id in &op.inputs {
                    // 检查输入是否已经是 AMP dtype
                    if let Some(value) = graph.values.get(&in_id) {
                        if value.ty.dtype == amp_dtype {
                            new_inputs.push(in_id);
                            continue;
                        }
                    }

                    // 插入 Cast
                    let cast_out_id = next_id;
                    next_id += 1;

                    let cast_op = Op {
                        id: cast_out_id,
                        name: format!("cast_to_amp_{}", in_id),
                        op_type: "cast".to_string(),
                        inputs: vec![in_id],
                        outputs: vec![cast_out_id],
                        attrs: {
                            let mut attrs = std::collections::HashMap::new();
                            attrs.insert(
                                "dtype".to_string(),
                                AttrValue::String(format!("{:?}", amp_dtype)),
                            );
                            attrs
                        },
                    };

                    // 创建输出 Value
                    let out_shape = graph
                        .values
                        .get(&in_id)
                        .map(|v| v.ty.shape.clone())
                        .unwrap_or(vec![]);

                    graph.values.insert(
                        cast_out_id,
                        crate::ir::dag::Value {
                            id: cast_out_id,
                            name: format!("cast_to_amp_{}", in_id),
                            ty: TensorType {
                                dtype: amp_dtype,
                                shape: out_shape,
                            },
                            producer: Some(cast_out_id),
                            scale: None,
                            zero_point: None,
                        },
                    );

                    new_ops.push(cast_op);
                    new_inputs.push(cast_out_id);
                }

                // 为输出插入 Cast (AMP dtype → FP32)
                let mut new_outputs = Vec::new();
                for &out_id in &op.outputs {
                    let cast_out_id = next_id;
                    next_id += 1;

                    let cast_op = Op {
                        id: cast_out_id,
                        name: format!("cast_from_amp_{}", out_id),
                        op_type: "cast".to_string(),
                        inputs: vec![out_id],
                        outputs: vec![cast_out_id],
                        attrs: {
                            let mut attrs = std::collections::HashMap::new();
                            attrs.insert(
                                "dtype".to_string(),
                                AttrValue::String("F32".to_string()),
                            );
                            attrs
                        },
                    };

                    let out_shape = graph
                        .values
                        .get(&out_id)
                        .map(|v| v.ty.shape.clone())
                        .unwrap_or(vec![]);

                    graph.values.insert(
                        cast_out_id,
                        crate::ir::dag::Value {
                            id: cast_out_id,
                            name: format!("cast_from_amp_{}", out_id),
                            ty: TensorType {
                                dtype: DataType::F32,
                                shape: out_shape,
                            },
                            producer: Some(cast_out_id),
                            scale: None,
                            zero_point: None,
                        },
                    );

                    new_ops.push(cast_op);
                    new_outputs.push(cast_out_id);
                }

                // 更新算子 (使用 AMP dtype)
                let mut new_op = op.clone();
                new_op.inputs = new_inputs;
                new_op.outputs = new_outputs;
                new_op.id = next_id;
                next_id += 1;
                new_ops.push(new_op);

                to_remove.push(op_id);
            } else {
                // 不转换的算子保持原样
                new_ops.push(op);
            }
        }

        // 更新参数 (参数也需要转换为 AMP dtype)
        // 但训练时参数保持在 FP32，forward 时动态转换
        // 这里只标记，实际转换在 forward 时做

        // 替换 graph
        for id in to_remove {
            graph.ops.remove(&id);
        }

        for op in new_ops {
            graph.ops.insert(op.id, op);
        }

        graph.next_id = next_id;

        // 更新输出 (确保输出是 FP32)
        let mut new_outputs = Vec::new();
        for &out_id in &graph.outputs {
            let out_value = graph.values.get(&out_id);
            if let Some(v) = out_value {
                if v.ty.dtype != DataType::F32 {
                    // 插入 Cast
                    let cast_out_id = next_id;
                    next_id += 1;

                    let cast_op = Op {
                        id: cast_out_id,
                        name: format!("cast_output_{}", out_id),
                        op_type: "cast".to_string(),
                        inputs: vec![out_id],
                        outputs: vec![cast_out_id],
                        attrs: {
                            let mut attrs = std::collections::HashMap::new();
                            attrs.insert(
                                "dtype".to_string(),
                                AttrValue::String("F32".to_string()),
                            );
                            attrs
                        },
                    };

                    let out_shape = v.ty.shape.clone();
                    graph.values.insert(
                        cast_out_id,
                        crate::ir::dag::Value {
                            id: cast_out_id,
                            name: format!("cast_output_{}", out_id),
                            ty: TensorType {
                                dtype: DataType::F32,
                                shape: out_shape,
                            },
                            producer: Some(cast_out_id),
                            scale: None,
                            zero_point: None,
                        },
                    );

                    graph.ops.insert(cast_out_id, cast_op);
                    new_outputs.push(cast_out_id);
                } else {
                    new_outputs.push(out_id);
                }
            }
        }
        graph.outputs = new_outputs;
        graph.next_id = next_id;

        Ok(())
    }

    fn should_run_in_amp(op_type: &str) -> bool {
        // 这些算子在 AMP dtype 下运行
        // 数值稳定的算子
        match op_type {
            // 数学
            "add" | "sub" | "mul" | "div" | "pow" |
            "exp" | "sqrt" | "log" | "abs" | "neg" |
            // 激活
            "relu" | "gelu" | "silu" | "sigmoid" | "tanh" |
            // 线性代数
            "matmul" | "batch_matmul" | "transpose" |
            // 卷积
            "conv2d" | "conv1d" | "conv3d" |
            // 池化
            "maxpool2d" | "avgpool2d" |
            // 张量操作
            "reshape" | "flatten" | "concat" | "slice" |
            // 归约
            "sum" | "mean" | "max" | "min" |
            // 嵌入
            "embedding" |
            // 注意力
            "scaled_dot_product_attention" |
            // 归一化
            "layer_norm" | "rms_norm" => true,

            // 这些算子保持 FP32 (数值敏感)
            "batch_norm" | "softmax" | "log_softmax" => false,
            "cross_entropy" | "mse" | "l1" | "bce" => false,
            _ => true,
        }
    }
}

// ============================================================
// Tensor 转换工具
// ============================================================

#[allow(unused_assignments)]
pub fn cast_to_dtype<T: crate::dtype::DType + Send + Sync>(
    input: &Tensor<f32>,
    _dtype: DataType,
) -> Tensor<f32> {
    // 这里用 cast 算子实现
    // 实际转换由 cast 算子处理
    // 由于我们使用 f32 存储所有值，cast 只改变 dtype 标记
    let mut result = input.clone();
    // 标记 dtype
    result = Tensor::new(input.data().to_vec(), input.shape());
    result
}

pub fn cast_from_amp<T: crate::dtype::DType + Send + Sync>(
    input: &Tensor<f32>,
) -> Tensor<f32> {
    input.clone()
}
