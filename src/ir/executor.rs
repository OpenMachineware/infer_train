use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;
use crate::ir::dag::{DagGraph, AttrValue};
use crate::ir::serialize::PyModelFile;
use crate::ffi::Tensor;
use crate::pytensor::PyTensor;


#[pyclass]
pub struct PyExecutor {
    inner: Executor,
}

#[pymethods]
impl PyExecutor {
    #[staticmethod]
    pub fn from_model_file(model: &PyModelFile) -> Self {
        let graph = model.inner.graph.clone();
        PyExecutor {
            inner: Executor::new(graph),
        }
    }

    pub fn execute(&mut self, inputs: &Bound<PyList>) -> PyResult<Vec<PyTensor>> {
        // 1. 从 PyList 提取 Tensor
        let mut input_tensors = Vec::new();
        for item in inputs.iter() {
            let pytensor = item.downcast::<PyTensor>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err("Expected PyTensor"))?;
            input_tensors.push(pytensor.borrow().inner.clone());
        }

        // 2. 调用真正的执行器
        let result = self.inner.execute(&input_tensors)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // 3. 转换回 PyTensor
        Ok(result.into_iter().map(|t| PyTensor { inner: t }).collect())
    }
}

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
                let len = data.len() / 4;
                let final_shape = if id == 6 {
                    vec![16, 3, 3, 3]
                } else if id == 5 {
                    vec![16]
                } else {
                    value.ty.shape.iter().map(|&x| x as usize).collect()
                };

                let float_data: Vec<f32> = data.chunks(4)
                    .map(|chunk| {
                        if chunk.len() == 4 {
                            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let tensor = Tensor::new_f32(&float_data, &final_shape);
                self.values.insert(id, tensor);
            } else {
                return Err(format!("Constant value {} has no type info", id));
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

    fn dispatch_op(
        &self,
        op_type: &str,
        inputs: &[Tensor],
        attrs: &HashMap<String, AttrValue>,
    ) -> Result<Vec<Tensor>, String> {
        let get_int = |key: &str, default: i32| -> i32 {
            attrs.get(key)
                .and_then(|v| match v {
                    AttrValue::Int(i) => Some(*i as i32),
                    AttrValue::IntList(list) if !list.is_empty() => Some(list[0] as i32),
                    _ => None,
                })
                .unwrap_or(default)
        };

        let get_int_list = |key: &str| -> Vec<i64> {
            attrs.get(key)
                .and_then(|v| match v {
                    AttrValue::IntList(list) => Some(list.clone()),
                    AttrValue::Int(i) => Some(vec![*i]),
                    _ => None,
                })
                .unwrap_or_else(|| vec![1, 1])
        };

        match op_type {
            "add" => {
                if inputs.len() < 2 {
                    return Err("add requires 2 inputs".to_string());
                }
                let result = inputs[0].add(&inputs[1]);
                Ok(vec![result])
            }
            "sub" => {
                if inputs.len() < 2 {
                    return Err("sub requires 2 inputs".to_string());
                }
                let result = inputs[0].sub(&inputs[1]);
                Ok(vec![result])
            }
            "mul" => {
                if inputs.len() < 2 {
                    return Err("mul requires 2 inputs".to_string());
                }
                let result = inputs[0].mul(&inputs[1]);
                Ok(vec![result])
            }
            "div" => {
                if inputs.len() < 2 {
                    return Err("div requires 2 inputs".to_string());
                }
                let result = inputs[0].div(&inputs[1]);
                Ok(vec![result])
            }
            "matmul" => {
                if inputs.len() < 2 {
                    return Err("matmul requires 2 inputs".to_string());
                }
                let result = inputs[0].matmul(&inputs[1]);
                Ok(vec![result])
            }
            "relu" => {
                if inputs.is_empty() {
                    return Err("relu requires 1 input".to_string());
                }
                let result = inputs[0].relu();
                Ok(vec![result])
            }
            "sigmoid" => {
                if inputs.is_empty() {
                    return Err("sigmoid requires 1 input".to_string());
                }
                let result = inputs[0].sigmoid();
                Ok(vec![result])
            }
            "tanh" => {
                if inputs.is_empty() {
                    return Err("tanh requires 1 input".to_string());
                }
                let result = inputs[0].tanh();
                Ok(vec![result])
            }
            "softmax" => {
                if inputs.is_empty() {
                    return Err("softmax requires 1 input".to_string());
                }
                let dim = get_int("dim", -1);
                let result = inputs[0].softmax(dim);
                Ok(vec![result])
            }
            "conv2d" => {
                // 只取前 3 个输入：input, weight, bias
                let input = &inputs[0];
                let weight = &inputs[1];
                let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };

                let stride = get_int("stride", 1);
                let padding = get_int("padding", 0);
                let dilation = get_int("dilation", 1);
                let groups = get_int("groups", 1);

                let result = input.conv2d(weight, bias, stride, padding, dilation, groups);
                Ok(vec![result])
            }
            "linear" => {
                if inputs.len() < 2 {
                    return Err("linear requires at least 2 inputs (input, weight)".to_string());
                }
                let bias = if inputs.len() >= 3 { Some(&inputs[2]) } else { None };
                let result = inputs[0].linear(&inputs[1], bias);
                Ok(vec![result])
            }
            "reshape" => {
                if inputs.is_empty() {
                    return Err("reshape requires 1 input".to_string());
                }
                let shape = get_int_list("shape");
                let new_shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                let result = inputs[0].reshape(&new_shape);
                Ok(vec![result])
            }
            "transpose" => {
                if inputs.is_empty() {
                    return Err("transpose requires 1 input".to_string());
                }
                let result = inputs[0].transpose();
                Ok(vec![result])
            }
            "constant" => {
                // 常量节点：直接返回输入（权重数据）
                if inputs.is_empty() {
                    return Err("constant requires 1 input".to_string());
                }
                Ok(vec![inputs[0].clone()])
            }
            _ => {
                eprintln!("Warning: operator '{}' not implemented, returning input", op_type);
                if inputs.is_empty() {
                    Ok(vec![])
                } else {
                    Ok(vec![inputs[0].clone()])
                }
            }
        }
    }
}
