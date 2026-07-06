// src/executor/executor.rs

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::ir::dag::{DagGraph, DataType, TensorType, AttrValue};
use crate::pytensor::PyTensor;
use crate::ffi::Tensor;

use super::math;
use super::nn;
use super::activation;
use super::tensor;
use super::index;
use super::control;
use super::quantized;

// ============================================================
// 真正的执行器
// ============================================================
pub struct Executor {
    graph: DagGraph,
    values: HashMap<u64, Tensor>,
}

impl Executor {
    pub fn new(graph: DagGraph) -> Self {
        Executor {
            graph,
            values: HashMap::new(),
        }
    }

    pub fn execute(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.values.clear();

        // 加载输入
        if inputs.len() != self.graph.inputs.len() {
            return Err(format!(
                "Expected {} inputs, got {}",
                self.graph.inputs.len(),
                inputs.len()
            ));
        }

        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            self.values.insert(input_id, inputs[i].clone());
        }

        // 加载权重（constants）
        for (&id, data) in &self.graph.constants {
            if let Some(value) = self.graph.values.get(&id) {
                let dtype = value.ty.dtype;
                let shape: Vec<usize> = value.ty.shape.iter().map(|&x| x as usize).collect();

                let tensor = match dtype {
                    DataType::F32 => {
                        let float_data: Vec<f32> = data.chunks(4)
                            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .collect();
                        Tensor::new_f32(&float_data, &shape)
                    }
                    DataType::F64 => {
                        let double_data: Vec<f64> = data.chunks(8)
                            .map(|chunk| f64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]))
                            .collect();
                        Tensor::new_f64(&double_data, &shape)
                    }
                    DataType::F16 => {
                        let u16_data: Vec<u16> = data.chunks(2)
                            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                            .collect();
                        Tensor::new_f16(&u16_data, &shape)
                    }
                    DataType::BF16 => {
                        let u16_data: Vec<u16> = data.chunks(2)
                            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                            .collect();
                        Tensor::new_bf16(&u16_data, &shape)
                    }
                    DataType::I8 => {
                        let i8_data: Vec<i8> = data.iter().map(|&b| b as i8).collect();
                        let scale = value.scale.unwrap_or(1.0);
                        let zero_point = value.zero_point.unwrap_or(0.0);
                        Tensor::new_quantized(&i8_data, &shape, scale, zero_point)
                    }
                    _ => {
                        return Err(format!("Unsupported dtype for weight: {:?}", dtype));
                    }
                };
                self.values.insert(id, tensor);
            }
        }

        // 拓扑排序 + 执行
        let order = self.graph.topological_sort()?;
        for op_id in order {
            let op = self.graph.get_op(op_id)
                .ok_or_else(|| format!("Op {} not found", op_id))?
                .clone();

            self.execute_op(op_id, &op)?;
        }

        // 收集输出
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

    fn execute_op(&mut self, _op_id: u64, op: &crate::ir::dag::Op) -> Result<(), String> {
        // 根据算子类型决定需要哪些输入
        let needed_indices = match op.op_type.as_str() {
            "conv2d" => vec![0, 1, 2],  // 只需要前 3 个：input, weight, bias
            "linear" => vec![0, 1, 2],  // input, weight, bias
            "add" | "sub" | "mul" | "div" | "matmul" => vec![0, 1],
            "relu" | "sigmoid" | "tanh" | "softmax" | "reshape" | "transpose" => vec![0],
            _ => {
                // 默认：所有输入都需要
                (0..op.inputs.len()).collect()
            }
        };

        let mut input_tensors = Vec::new();
        for &idx in &needed_indices {
            if idx < op.inputs.len() {
                let in_id = op.inputs[idx];
                if let Some(t) = self.values.get(&in_id) {
                    input_tensors.push(t.clone());
                } else {
                    return Err(format!("Input value {} not found for op {}", in_id, op.id));
                }
            }
        }

        let outputs = self.dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                self.values.insert(out_id, outputs[i].clone());
            }
        }

        Ok(())
    }

    fn dispatch_op(&self, op_type: &str, inputs: &[Tensor], attrs: &HashMap<String, crate::ir::dag::AttrValue>) -> Result<Vec<Tensor>, String> {
        // 按类别分派到子模块 先检查是否是量化算子
        if op_type.starts_with("quantized_") {
            return quantized::dispatch_quantized(op_type, inputs, attrs);
        }
        if let Ok(result) = math::dispatch_math(op_type, inputs, attrs) {
            return Ok(result);
        }
        if let Ok(result) = nn::dispatch_nn(op_type, inputs, attrs) {
            return Ok(result);
        }
        if let Ok(result) = activation::dispatch_activation(op_type, inputs, attrs) {
            return Ok(result);
        }
        if let Ok(result) = tensor::dispatch_tensor(op_type, inputs, attrs) {
            return Ok(result);
        }
        if let Ok(result) = index::dispatch_index(op_type, inputs, attrs) {
            return Ok(result);
        }
        if let Ok(result) = control::dispatch_control(op_type, inputs, attrs) {
            return Ok(result);
        }
        Err(format!("Unknown operator: {}", op_type))
    }
}

// ============================================================
// Python 绑定
// ============================================================
#[pyclass]
pub struct PyExecutor {
    inner: Executor,
}

#[pymethods]
impl PyExecutor {
    #[staticmethod]
    pub fn from_model_file(model: &crate::ir::serialize::PyModelFile) -> Self {
        let graph = model.inner.graph.clone();
        PyExecutor {
            inner: Executor::new(graph),
        }
    }

    pub fn execute(&mut self, inputs: &Bound<PyList>) -> PyResult<Vec<PyTensor>> {
        let mut input_tensors = Vec::new();
        for item in inputs.iter() {
            let pytensor = item.downcast::<PyTensor>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err("Expected PyTensor"))?;
            input_tensors.push(pytensor.borrow().inner.clone());
        }

        let result = self.inner.execute(&input_tensors)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(result.into_iter().map(|t| PyTensor { inner: t }).collect())
    }
}
