// src/autograd/engine.rs

use std::collections::HashMap;

use crate::autograd::tape::TapeEntry;
use crate::executor::Executor;
use crate::ir::dag::{DagGraph, Op};
use crate::tensor::Tensor;

use super::backward::backward;
use super::tape::Tape;

#[derive(Debug, Clone)]
pub struct AutogradConfig {
    pub enable_grad: bool,
    pub retain_graph: bool,
}

impl Default for AutogradConfig {
    fn default() -> Self {
        AutogradConfig { enable_grad: true, retain_graph: false }
    }
}

// ============================================================
// Automatic Differentiation Engine
// ============================================================

pub struct AutogradEngine {
    graph: DagGraph,
    #[allow(dead_code)]
    executor: Executor,
    tape: Tape,
    values: HashMap<u64, Tensor<f32>>,
    grads: HashMap<u64, Tensor<f32>>,
    config: AutogradConfig,
    param_ids: Vec<u64>,
}

impl AutogradEngine {
    pub fn new(graph: DagGraph, param_ids: Vec<u64>) -> Self {
        AutogradEngine {
            graph: graph.clone(),
            executor: Executor::new(graph),
            tape: Tape::new(),
            values: HashMap::new(),
            grads: HashMap::new(),
            config: AutogradConfig::default(),
            param_ids,
        }
    }

    pub fn with_config(mut self, config: AutogradConfig) -> Self {
        self.config = config;
        self
    }

    // ============================================================
    // Forward Propagation (Record Computation Graph)
    // ============================================================

    pub fn forward(
        &mut self,
        inputs: &[Tensor<f32>],
    ) -> Result<Vec<Tensor<f32>>, String> {
        self.values.clear();
        self.tape = Tape::new();
        self.grads.clear();

        // Register inputs
        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            if i < inputs.len() {
                self.values.insert(input_id, inputs[i].clone());
                self.tape.register_input(input_id);
            }
        }

        // Register parameters
        for &param_id in &self.param_ids {
            if let Some(value) = self.graph.values.get(&param_id) {
                if let Some(data) = self.graph.constants.get(&param_id) {
                    let shape: Vec<usize> = value
                        .ty
                        .shape
                        .iter()
                        .map(|&x| if x == -1 { 0 } else { x as usize })
                        .collect();
                    let tensor = Self::bytes_to_tensor(data, &shape)?;
                    self.values.insert(param_id, tensor);
                    self.tape.register_parameter(param_id);
                }
            }
        }

        // Register constants
        for (&id, data) in &self.graph.constants {
            if !self.param_ids.contains(&id) {
                if let Some(value) = self.graph.values.get(&id) {
                    let shape: Vec<usize> = value
                        .ty
                        .shape
                        .iter()
                        .map(|&x| if x == -1 { 0 } else { x as usize })
                        .collect();
                    let tensor = Self::bytes_to_tensor(data, &shape)?;
                    self.values.insert(id, tensor);
                    self.tape.register_constant(id);
                }
            }
        }

        // Execute topological sort, record each operator
        let order = self.graph.topological_sort()?;
        for op_id in order {
            let op = self
                .graph
                .get_op(op_id)
                .ok_or_else(|| format!("Op {} not found", op_id))?
                .clone();
            self.execute_op_and_record(&op)?;
        }

        // Collect outputs
        let mut result = Vec::new();
        for &out_id in &self.graph.outputs {
            if let Some(t) = self.values.get(&out_id) {
                result.push(t.clone());
            } else {
                return Err(format!("Output value {} not found", out_id));
            }
        }

        Ok(result)
    }

    // ============================================================
    // Execute Operator and Record to Tape
    // ============================================================

    fn execute_op_and_record(&mut self, op: &Op) -> Result<(), String> {
        let mut input_tensors = Vec::new();
        for &in_id in &op.inputs {
            if let Some(t) = self.values.get(&in_id) {
                input_tensors.push(t.clone());
            } else {
                return Err(format!(
                    "Input value {} not found for op {}",
                    in_id, op.id
                ));
            }
        }

        // Execute operator
        let outputs = crate::executor::dispatch_op(
            &op.op_type,
            &input_tensors,
            &op.attrs,
        )?;

        // Store output and record to Tape
        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                self.values.insert(out_id, outputs[i].clone());

                // Create TapeEntry based on op_type
                let entry = self.create_tape_entry(
                    &op,
                    &input_tensors,
                    &outputs,
                    out_id,
                );
                if let Some(e) = entry {
                    self.tape.push(e);
                }
            }
        }

        Ok(())
    }

    // ============================================================
    // Create TapeEntry
    // ============================================================

    fn create_tape_entry(
        &self,
        op: &Op,
        inputs: &[Tensor<f32>],
        outputs: &[Tensor<f32>],
        out_id: u64,
    ) -> Option<TapeEntry> {
        let input_ids = &op.inputs;

        if inputs.is_empty() || outputs.is_empty() {
            return None;
        }

        match op.op_type.as_str() {
            "add" => Some(TapeEntry::Add {
                input_a: input_ids[0],
                input_b: input_ids[1],
                output: out_id,
            }),
            "sub" => Some(TapeEntry::Sub {
                input_a: input_ids[0],
                input_b: input_ids[1],
                output: out_id,
            }),
            "mul" => Some(TapeEntry::Mul {
                input_a: input_ids[0],
                input_b: input_ids[1],
                output: out_id,
            }),
            "div" => Some(TapeEntry::Div {
                input_a: input_ids[0],
                input_b: input_ids[1],
                output: out_id,
            }),
            "exp" => {
                Some(TapeEntry::Exp { input: input_ids[0], output: out_id })
            }
            "sqrt" => {
                Some(TapeEntry::Sqrt { input: input_ids[0], output: out_id })
            }
            "log" => {
                Some(TapeEntry::Log { input: input_ids[0], output: out_id })
            }
            "neg" => {
                Some(TapeEntry::Neg { input: input_ids[0], output: out_id })
            }
            "relu" => {
                Some(TapeEntry::Relu { input: input_ids[0], output: out_id })
            }
            "sigmoid" => {
                Some(TapeEntry::Sigmoid { input: input_ids[0], output: out_id })
            }
            "tanh" => {
                Some(TapeEntry::Tanh { input: input_ids[0], output: out_id })
            }
            "matmul" => Some(TapeEntry::MatMul {
                input_a: input_ids[0],
                input_b: input_ids[1],
                output: out_id,
            }),
            "reshape" => Some(TapeEntry::Reshape {
                input: input_ids[0],
                output: out_id,
                new_shape: outputs[0].shape().to_vec(),
            }),
            "concat" => Some(TapeEntry::Concat {
                inputs: input_ids.clone(),
                dim: op
                    .attrs
                    .get("dim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0),
                output: out_id,
            }),
            "pow" => {
                let exponent = attrs_get_float(&op.attrs, "exponent", 2.0);
                Some(TapeEntry::Pow {
                    input: input_ids[0],
                    exponent,
                    output: out_id,
                })
            }
            "softmax" => Some(TapeEntry::Softmax {
                input: input_ids[0],
                dim: op
                    .attrs
                    .get("dim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0),
                output: out_id,
            }),
            "sum" => Some(TapeEntry::ReduceSum {
                input: input_ids[0],
                dim: op
                    .attrs
                    .get("dim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0),
                keepdim: op
                    .attrs
                    .get("keepdim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false),
                output: out_id,
            }),
            "mean" => Some(TapeEntry::ReduceMean {
                input: input_ids[0],
                dim: op
                    .attrs
                    .get("dim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0),
                keepdim: op
                    .attrs
                    .get("keepdim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false),
                output: out_id,
            }),
            "slice" => Some(TapeEntry::Slice {
                input: input_ids[0],
                dim: op
                    .attrs
                    .get("dim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0),
                start: op
                    .attrs
                    .get("start")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i),
                        _ => None,
                    })
                    .unwrap_or(0),
                end: op
                    .attrs
                    .get("end")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i),
                        _ => None,
                    })
                    .unwrap_or(-1),
                step: op
                    .attrs
                    .get("step")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i),
                        _ => None,
                    })
                    .unwrap_or(1),
                output: out_id,
            }),
            "squeeze" => Some(TapeEntry::Squeeze {
                input: input_ids[0],
                dim: op.attrs.get("dim").and_then(|v| match v {
                    crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
                    _ => None,
                }),
                output: out_id,
            }),
            "unsqueeze" => Some(TapeEntry::Unsqueeze {
                input: input_ids[0],
                dim: op
                    .attrs
                    .get("dim")
                    .and_then(|v| match v {
                        crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(0),
                output: out_id,
            }),
            "select" => Some(TapeEntry::Select {
                condition: input_ids[0],
                true_val: input_ids[1],
                false_val: input_ids[2],
                output: out_id,
            }),
            "embedding" => Some(TapeEntry::Embedding {
                indices: input_ids[0],
                weight: input_ids[1],
                output: out_id,
            }),
            "conv2d" => {
                let stride = attrs_get_int(&op.attrs, "stride", 1);
                let padding = attrs_get_int(&op.attrs, "padding", 0);
                let dilation = attrs_get_int(&op.attrs, "dilation", 1);
                let groups = attrs_get_int(&op.attrs, "groups", 1);
                let bias = if input_ids.len() >= 3 {
                    Some(input_ids[2])
                } else {
                    None
                };
                Some(TapeEntry::Conv2d {
                    input: input_ids[0],
                    weight: input_ids[1],
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    output: out_id,
                })
            }
            "max_pool" => {
                let kernel_size = attrs_get_int(&op.attrs, "kernel_size", 2);
                let stride = attrs_get_int(&op.attrs, "stride", kernel_size);
                let padding = attrs_get_int(&op.attrs, "padding", 0);
                Some(TapeEntry::MaxPool {
                    input: input_ids[0],
                    kernel_size,
                    stride,
                    padding,
                    output: out_id,
                })
            }
            "avg_pool" => {
                let kernel_size = attrs_get_int(&op.attrs, "kernel_size", 2);
                let stride = attrs_get_int(&op.attrs, "stride", kernel_size);
                let padding = attrs_get_int(&op.attrs, "padding", 0);
                Some(TapeEntry::AvgPool {
                    input: input_ids[0],
                    kernel_size,
                    stride,
                    padding,
                    output: out_id,
                })
            }
            "batch_norm" => {
                let eps = attrs_get_float(&op.attrs, "eps", 1e-5);
                Some(TapeEntry::BatchNorm {
                    input: input_ids[0],
                    weight: input_ids[1],
                    bias: input_ids[2],
                    running_mean: input_ids[3],
                    running_var: input_ids[4],
                    eps,
                    output: out_id,
                })
            }
            "layer_norm" => {
                let eps = attrs_get_float(&op.attrs, "eps", 1e-5);
                Some(TapeEntry::LayerNorm {
                    input: input_ids[0],
                    weight: input_ids[1],
                    bias: input_ids[2],
                    eps,
                    output: out_id,
                })
            }
            _ => {
                // For unrecorded operators,
                // don't add to tape (but keep forward result)
                None
            }
        }
    }

    // ============================================================
    // Backward Propagation
    // ============================================================

    pub fn backward(
        &mut self,
        loss: &Tensor<f32>,
    ) -> Result<&HashMap<u64, Tensor<f32>>, String> {
        if !self.config.enable_grad {
            return Ok(&self.grads);
        }

        self.grads = backward(loss, &self.tape, &self.values);

        Ok(&self.grads)
    }

    // ============================================================
    // Get Gradient
    // ============================================================

    pub fn get_grad(&self, param_id: u64) -> Option<&Tensor<f32>> {
        self.grads.get(&param_id)
    }

    pub fn get_all_grads(&self) -> &HashMap<u64, Tensor<f32>> {
        &self.grads
    }

    // Enable gradient tracking
    pub fn set_requires_grad(&mut self, param_id: u64, requires_grad: bool) {
        if requires_grad && !self.param_ids.contains(&param_id) {
            self.param_ids.push(param_id);
        } else if !requires_grad {
            self.param_ids.retain(|&id| id != param_id);
        }
    }

    // Check if parameter requires gradient
    pub fn requires_grad(&self, param_id: u64) -> bool {
        self.param_ids.contains(&param_id)
    }

    // Gradient accumulation
    pub fn accumulate_grad(&mut self, param_id: u64, grad: Tensor<f32>) {
        self.grads
            .entry(param_id)
            .and_modify(|existing| {
                // Accumulate gradient
                let existing_data = existing.data_mut();
                let grad_data = grad.data();
                for i in 0..existing_data.len().min(grad_data.len()) {
                    existing_data[i] += grad_data[i];
                }
            })
            .or_insert(grad);
    }

    // Zero gradients
    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }

    // Gradient clipping (prevent gradient explosion)
    pub fn clip_grad(&mut self, max_norm: f32) {
        let mut total_norm = 0.0;
        for grad in self.grads.values() {
            for &v in grad.data() {
                total_norm += v * v;
            }
        }
        total_norm = total_norm.sqrt();
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for grad in self.grads.values_mut() {
                for v in grad.data_mut() {
                    *v *= scale;
                }
            }
        }
    }

    // Gradient checking (for verification)
    pub fn grad_check(
        &self,
        param_id: u64,
        grad: &Tensor<f32>,
        eps: f32,
    ) -> bool {
        if let Some(computed_grad) = self.grads.get(&param_id) {
            let computed = computed_grad.data();
            let numeric = grad.data();
            if computed.len() != numeric.len() {
                return false;
            }
            let mut max_diff = 0.0;
            for i in 0..computed.len() {
                let diff = (computed[i] - numeric[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            max_diff < eps
        } else {
            false
        }
    }

    // ============================================================
    // Utility Functions
    // ============================================================

    fn bytes_to_tensor(
        data: &[u8],
        shape: &[usize],
    ) -> Result<Tensor<f32>, String> {
        let float_data: Vec<f32> = data
            .chunks(4)
            .map(|chunk| {
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            })
            .collect();
        Ok(Tensor::new(float_data, shape))
    }

    pub fn param_ids(&self) -> &[u64] {
        &self.param_ids
    }
}

// ============================================================
// Helper Functions
// ============================================================

fn attrs_get_int(
    attrs: &std::collections::HashMap<String, crate::ir::dag::AttrValue>,
    key: &str,
    default: usize,
) -> usize {
    attrs
        .get(key)
        .and_then(|v| match v {
            crate::ir::dag::AttrValue::Int(i) => Some(*i as usize),
            crate::ir::dag::AttrValue::IntList(list) if !list.is_empty() => {
                Some(list[0] as usize)
            }
            _ => None,
        })
        .unwrap_or(default)
}

fn attrs_get_float(
    attrs: &std::collections::HashMap<String, crate::ir::dag::AttrValue>,
    key: &str,
    default: f32,
) -> f32 {
    attrs
        .get(key)
        .and_then(|v| match v {
            crate::ir::dag::AttrValue::Float(f) => Some(*f as f32),
            crate::ir::dag::AttrValue::Int(i) => Some(*i as f32),
            _ => None,
        })
        .unwrap_or(default)
}
