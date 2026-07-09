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
// AMP Graph Converter
// ============================================================

pub struct AmpGraphConverter;

impl AmpGraphConverter {
    /// Convert DAG to mixed precision DAG
    /// Rules:
    /// 1. Insert Cast at input (FP32 → AMP dtype)
    /// 2. Insert Cast at output (AMP dtype → FP32)
    /// 3. Automatically adapt input/output of each operator to AMP dtype
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

        // Collect all nodes to convert
        let op_ids: Vec<u64> = graph.ops.keys().cloned().collect();

        for &op_id in &op_ids {
            let op = match graph.ops.get(&op_id) {
                Some(o) => o.clone(),
                None => continue,
            };

            // Determine if operator should run in AMP dtype
            if Self::should_run_in_amp(&op.op_type) {
                // Insert Cast for each input (FP32 → AMP dtype)
                let mut new_inputs = Vec::new();
                for &in_id in &op.inputs {
                    // Check if input is already AMP dtype
                    if let Some(value) = graph.values.get(&in_id) {
                        if value.ty.dtype == amp_dtype {
                            new_inputs.push(in_id);
                            continue;
                        }
                    }

                    // Insert Cast
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

                    // Create output Value
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

                // Insert Cast for output (AMP dtype → FP32)
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

                // Update operator (use AMP dtype)
                let mut new_op = op.clone();
                new_op.inputs = new_inputs;
                new_op.outputs = new_outputs;
                new_op.id = next_id;
                next_id += 1;
                new_ops.push(new_op);

                to_remove.push(op_id);
            } else {
                // Operators that don't convert remain unchanged
                new_ops.push(op);
            }
        }

        // Update parameters (parameters also need to be converted to AMP dtype)
        // But during training, parameters stay in FP32 and are converted
        // dynamically during forward
        // Only mark here, actual conversion happens during forward

        // Replace graph
        for id in to_remove {
            graph.ops.remove(&id);
        }

        for op in new_ops {
            graph.ops.insert(op.id, op);
        }

        graph.next_id = next_id;

        // Update outputs (ensure outputs are FP32)
        let mut new_outputs = Vec::new();
        for &out_id in &graph.outputs {
            let out_value = graph.values.get(&out_id);
            if let Some(v) = out_value {
                if v.ty.dtype != DataType::F32 {
                    // Insert Cast
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
        // These operators run in AMP dtype
        // Numerically stable operators
        match op_type {
            // Math
            "add" | "sub" | "mul" | "div" | "pow" |
            "exp" | "sqrt" | "log" | "abs" | "neg" |
            // Activation
            "relu" | "gelu" | "silu" | "sigmoid" | "tanh" |
            // Linear algebra
            "matmul" | "batch_matmul" | "transpose" |
            // Convolution
            "conv2d" | "conv1d" | "conv3d" |
            // Pooling
            "maxpool2d" | "avgpool2d" |
            // Tensor operations
            "reshape" | "flatten" | "concat" | "slice" |
            // Reduction
            "sum" | "mean" | "max" | "min" |
            // Embedding
            "embedding" |
            // Attention
            "scaled_dot_product_attention" |
            // Normalization
            "layer_norm" | "rms_norm" => true,

            // These operators stay in FP32 (numerically sensitive)
            "batch_norm" | "softmax" | "log_softmax" => false,
            "cross_entropy" | "mse" | "l1" | "bce" => false,
            _ => true,
        }
    }
}

// ============================================================
// Tensor Conversion Utilities
// ============================================================

#[allow(unused_assignments)]
pub fn cast_to_dtype<T: crate::dtype::DType + Send + Sync>(
    input: &Tensor<f32>,
    _dtype: DataType,
) -> Tensor<f32> {
    // Implemented using cast operator
    // Actual conversion handled by cast operator
    // Since we store all values as f32, cast only changes dtype tag
    let mut result = input.clone();
    // Mark dtype
    result = Tensor::new(input.data().to_vec(), input.shape());
    result
}

pub fn cast_from_amp<T: crate::dtype::DType + Send + Sync>(
    input: &Tensor<f32>,
) -> Tensor<f32> {
    input.clone()
}
