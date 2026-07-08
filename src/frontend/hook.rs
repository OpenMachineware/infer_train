// src/frontend/hook.rs

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::autograd::AutogradEngine;
use crate::ir::cfg::{CfgGraph, CfgOp};
use crate::ir::dag::{AttrValue, DagGraph, DataType};
use crate::ir::serialize::ModelFile;
use crate::transform::FullOptimizer;

// ============================================================
// 辅助函数（在 impl HookTracer 外面，避开 #[pymethods]）
// ============================================================

fn extract_value_id(py: Python<'_>, obj: &Py<PyAny>) -> Result<u64, String> {
    let obj = obj.bind(py);

    // 暂时注释 pytensor 依赖
    // if let Ok(pytensor) = obj.downcast::<crate::pytensor::PyTensor>() {
    //     let tensor = &pytensor.borrow().inner;
    //     return Ok(tensor as *const _ as u64);
    // }

    if let Ok(py_id) = obj.call_method0("__hash__") {
        if let Ok(id_val) = py_id.extract::<u64>() {
            return Ok(id_val);
        }
    }

    Ok(obj.as_ptr() as u64)
}

fn extract_tensor_info(
    py: Python<'_>,
    obj: &Py<PyAny>,
) -> Result<(DataType, Vec<i64>), String> {
    let obj = obj.bind(py);

    // 暂时注释 pytensor/ffi 依赖，直接通过 Python 属性获取
    // if let Ok(pytensor) = obj.downcast::<crate::pytensor::PyTensor>() {
    //     let tensor = &pytensor.borrow().inner;
    //     let dtype = tensor.dtype();
    //     let shape = tensor.shape();
    //
    //     let rust_dtype = match dtype {
    //         crate::ffi::it_dtype_t::IT_DTYPE_F32 => DataType::F32,
    //         crate::ffi::it_dtype_t::IT_DTYPE_F64 => DataType::F64,
    //         crate::ffi::it_dtype_t::IT_DTYPE_F16 => DataType::F16,
    //         crate::ffi::it_dtype_t::IT_DTYPE_BF16 => DataType::BF16,
    //         crate::ffi::it_dtype_t::IT_DTYPE_I8 => DataType::I8,
    //     };
    //
    //     let rust_shape: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    //     return Ok((rust_dtype, rust_shape));
    // }

    if let Ok(shape) = obj.call_method0("shape") {
        if let Ok(shape_list) = shape.extract::<Vec<i64>>() {
            let dtype_str = if let Ok(dtype) = obj.getattr("dtype") {
                if let Ok(str_val) = dtype.call_method0("__str__") {
                    str_val.extract::<String>().unwrap_or("f32".to_string())
                } else {
                    "f32".to_string()
                }
            } else {
                "f32".to_string()
            };

            let rust_dtype = match dtype_str.as_str() {
                s if s.contains("float32") || s.contains("f32") => {
                    DataType::F32
                }
                s if s.contains("float64") || s.contains("f64") => {
                    DataType::F64
                }
                s if s.contains("float16") || s.contains("f16") => {
                    DataType::F16
                }
                s if s.contains("bfloat16") || s.contains("bf16") => {
                    DataType::BF16
                }
                s if s.contains("int8") || s.contains("i8") => DataType::I8,
                s if s.contains("int32") || s.contains("i32") => DataType::I32,
                s if s.contains("int64") || s.contains("i64") => DataType::I64,
                s if s.contains("bool") => DataType::Bool,
                _ => DataType::F32,
            };

            return Ok((rust_dtype, shape_list));
        }
    }

    Ok((DataType::F32, Vec::new()))
}

fn convert_attrs(
    py: Python<'_>,
    attrs: &HashMap<String, PyObject>,
) -> Result<HashMap<String, AttrValue>, String> {
    let mut result = HashMap::new();

    for (key, value) in attrs {
        let val = value.bind(py);

        if let Ok(i) = val.extract::<i64>() {
            result.insert(key.clone(), AttrValue::Int(i));
        } else if let Ok(f) = val.extract::<f64>() {
            result.insert(key.clone(), AttrValue::Float(f));
        } else if let Ok(b) = val.extract::<bool>() {
            result.insert(key.clone(), AttrValue::Bool(b));
        } else if let Ok(s) = val.extract::<String>() {
            result.insert(key.clone(), AttrValue::String(s));
        } else if let Ok(list) = val.extract::<Vec<i64>>() {
            result.insert(key.clone(), AttrValue::IntList(list));
        } else if let Ok(list) = val.extract::<Vec<f64>>() {
            result.insert(key.clone(), AttrValue::FloatList(list));
        } else if let Ok(shape) = val.extract::<Vec<i64>>() {
            result.insert(key.clone(), AttrValue::Shape(shape));
        }
    }

    Ok(result)
}

// ============================================================
// HookTracer
// ============================================================

#[pyclass]
pub struct HookTracer {
    cfg: Arc<Mutex<CfgGraph>>,
    block_stack: Vec<u64>,
    current_block: u64,
    value_map: HashMap<u64, u64>,
    op_counter: u64,
    is_tracing: bool,
    block_counter: u64,
    model_name: String,
    framework: String,
    autograd: Option<AutogradEngine>,
    param_ids: Vec<u64>,
    training_mode: bool,
}

#[pymethods]
impl HookTracer {
    #[new]
    pub fn new() -> Self {
        let mut cfg = CfgGraph::new("traced_model");
        let entry = cfg.add_block("entry");
        cfg.set_entry(entry);

        HookTracer {
            cfg: Arc::new(Mutex::new(cfg)),
            block_stack: vec![entry],
            current_block: entry,
            value_map: HashMap::new(),
            op_counter: 0,
            is_tracing: false,
            block_counter: 1,
            model_name: "model".to_string(),
            framework: "torch".to_string(),
            autograd: None,
            param_ids: Vec::new(),
            training_mode: false,
        }
    }

    pub fn start_tracing(&mut self) {
        self.is_tracing = true;
        self.value_map.clear();
        self.op_counter = 0;
        self.block_counter = 1;
        println!("[HookTracer] Tracing started");
    }

    pub fn stop_tracing(&mut self) -> PyResult<()> {
        self.is_tracing = false;
        println!("[HookTracer] Tracing stopped");
        Ok(())
    }

    pub fn is_tracing(&self) -> bool {
        self.is_tracing
    }

    pub fn reset(&mut self) {
        let mut cfg = CfgGraph::new("traced_model");
        let entry = cfg.add_block("entry");
        cfg.set_entry(entry);
        *self.cfg.lock().unwrap() = cfg;
        self.block_stack = vec![entry];
        self.current_block = entry;
        self.value_map.clear();
        self.op_counter = 0;
        self.is_tracing = false;
        self.block_counter = 1;
        self.autograd = None;
        self.param_ids.clear();
        self.training_mode = false;
    }

    pub fn set_model_name(&mut self, name: String) {
        self.model_name = name;
    }

    pub fn set_framework(&mut self, framework: String) {
        self.framework = framework;
    }

    pub fn set_training_mode(&mut self, mode: bool) {
        self.training_mode = mode;
    }

    // ============================================================
    // 记录参数 (供 Python 补丁调用)
    // ============================================================

    pub fn register_parameter(&mut self, param_id: u64) {
        if !self.param_ids.contains(&param_id) {
            self.param_ids.push(param_id);
        }
    }

    pub fn get_param_ids(&self) -> Vec<u64> {
        self.param_ids.clone()
    }

    // ============================================================
    // 反向传播 (供 Python 补丁调用)
    // ============================================================

    pub fn backward(&mut self, loss: Py<PyAny>) -> PyResult<()> {
        if !self.training_mode {
            return Ok(());
        }

        // 获取 DAG
        let cfg = self.cfg.lock().unwrap().clone();
        let dag = FullOptimizer::optimize_full(&mut cfg.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // 创建或获取 AutogradEngine
        if self.autograd.is_none() {
            self.autograd =
                Some(AutogradEngine::new(dag, self.param_ids.clone()));
        }

        // 从 loss 提取 Tensor
        Python::with_gil(|py| {
            let _loss_obj = loss.bind(py);
            // 获取 loss 值 (简化版：假设是标量)
            // TODO: 从 PyTorch Tensor 提取数据
            Ok(())
        })
    }

    // ============================================================
    // 更新权重 (供 Python 补丁调用)
    // ============================================================

    pub fn step(&mut self) -> PyResult<()> {
        if !self.training_mode || self.autograd.is_none() {
            return Ok(());
        }

        // 使用 AutogradEngine 更新权重
        // TODO: 实现权重更新
        Ok(())
    }

    // ============================================================
    // 获取 DAG (供导出使用)
    // ============================================================

    pub fn get_dag(&self) -> PyResult<Py<PyAny>> {
        let cfg = self.cfg.lock().unwrap().clone();
        let dag = FullOptimizer::optimize_full(&mut cfg.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Python::with_gil(|py| {
            let py_dag = crate::ir::serialize::PyDagGraph { inner: dag };
            Ok(Py::new(py, py_dag)?.into_any())
        })
    }

    // ============================================================
    // 记录算子
    // ============================================================

    #[pyo3(signature = (op_type, inputs, outputs, attrs, name=None))]
    pub fn record_op(
        &mut self,
        op_type: String,
        inputs: Vec<Py<PyAny>>,
        outputs: Vec<Py<PyAny>>,
        attrs: HashMap<String, PyObject>,
        name: Option<String>,
    ) -> PyResult<()> {
        if !self.is_tracing {
            return Ok(());
        }

        Python::with_gil(|py| {
            // --- 提取 input IDs ---
            let mut input_ids = Vec::new();
            for obj in &inputs {
                let id = extract_value_id(py, obj).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(e)
                })?;
                input_ids.push(id);
            }

            // --- 提取 output IDs ---
            let mut output_ids = Vec::new();
            for obj in &outputs {
                let id = extract_value_id(py, obj).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(e)
                })?;
                output_ids.push(id);
            }

            // --- 转换 attrs ---
            let rust_attrs = convert_attrs(py, &attrs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            let op_id = self.op_counter;
            self.op_counter += 1;

            let op_name = name
                .clone()
                .unwrap_or_else(|| format!("{}_{}", op_type, op_id));

            let cfg_op = CfgOp {
                id: op_id,
                op_type: op_type.clone(),
                inputs: input_ids.clone(),
                outputs: output_ids.clone(),
                attrs: rust_attrs,
                name: op_name,
            };

            {
                let mut cfg = self.cfg.lock().unwrap();
                cfg.add_op(self.current_block, cfg_op).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(e)
                })?;
            }

            // --- 提取 tensor info ---
            for (i, out_id) in output_ids.iter().enumerate() {
                self.value_map.insert(*out_id, *out_id);
                if i < outputs.len() {
                    if let Ok((dtype, shape)) =
                        extract_tensor_info(py, &outputs[i])
                    {
                        let mut cfg = self.cfg.lock().unwrap();
                        cfg.value_types.insert(*out_id, (dtype, shape));
                    }
                }
            }

            Ok(())
        })
    }

    // ============================================================
    // 控制流
    // ============================================================

    #[pyo3(signature = (name=None))]
    pub fn begin_block(&mut self, name: Option<String>) -> PyResult<u64> {
        if !self.is_tracing {
            return Ok(0);
        }

        let block_name =
            name.unwrap_or_else(|| format!("block_{}", self.block_counter));
        self.block_counter += 1;

        let mut cfg = self.cfg.lock().unwrap();
        let block_id = cfg.add_block(&block_name);
        self.current_block = block_id;
        self.block_stack.push(block_id);

        Ok(block_id)
    }

    pub fn end_block(&mut self) -> PyResult<()> {
        if !self.is_tracing {
            return Ok(());
        }

        self.block_stack.pop();
        if let Some(&block_id) = self.block_stack.last() {
            self.current_block = block_id;
        }

        Ok(())
    }

    pub fn record_branch(
        &mut self,
        condition: Py<PyAny>,
        true_block: u64,
        false_block: u64,
    ) -> PyResult<()> {
        if !self.is_tracing {
            return Ok(());
        }

        let cond_id = Python::with_gil(|py| {
            extract_value_id(py, &condition)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })?;

        let merge_name = format!("merge_{}", self.block_counter);
        self.block_counter += 1;

        let mut cfg = self.cfg.lock().unwrap();
        let merge_id = cfg.add_block(&merge_name);

        cfg.set_branch(
            self.current_block,
            cond_id,
            true_block,
            false_block,
            merge_id,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        cfg.add_edge(self.current_block, true_block)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        cfg.add_edge(self.current_block, false_block)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        cfg.add_edge(true_block, merge_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        cfg.add_edge(false_block, merge_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        self.current_block = merge_id;
        self.block_stack.push(merge_id);

        Ok(())
    }

    // ============================================================
    // 高层 API
    // ============================================================

    pub fn trace_to_dag(
        &mut self,
        model: Py<PyAny>,
        inputs: Vec<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.start_tracing();

        let result = Python::with_gil(|py| {
            let model = model.bind(py);
            let args = PyTuple::new(py, inputs.iter().map(|x| x.bind(py)))?;
            let _outputs = model.call_method("forward", args, None)?;
            self.stop_tracing()?;
            Ok::<_, PyErr>(())
        });

        if let Err(e) = result {
            self.is_tracing = false;
            return Err(e);
        }

        let cfg = self.cfg.lock().unwrap().clone();
        let dag = FullOptimizer::optimize_full(&mut cfg.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Python::with_gil(|py| {
            let py_dag = crate::ir::serialize::PyDagGraph { inner: dag };
            Ok(Py::new(py, py_dag)?.into_any())
        })
    }

    #[pyo3(signature = (model, inputs, path, model_name=None))]
    pub fn trace_and_export(
        &mut self,
        model: Py<PyAny>,
        inputs: Vec<Py<PyAny>>,
        path: String,
        model_name: Option<String>,
    ) -> PyResult<()> {
        self.trace_and_export_with_config(
            model,
            inputs,
            path,
            model_name,
            false,
            "adam".to_string(),
            0.001,
        )
    }

    #[pyo3(signature = (model, inputs, path, model_name=None, optimizer="adam".to_string(), learning_rate=0.001))]
    pub fn trace_and_export_trainable(
        &mut self,
        model: Py<PyAny>,
        inputs: Vec<Py<PyAny>>,
        path: String,
        model_name: Option<String>,
        optimizer: String,
        learning_rate: f32,
    ) -> PyResult<()> {
        self.trace_and_export_with_config(
            model,
            inputs,
            path,
            model_name,
            true,
            optimizer,
            learning_rate,
        )
    }

    #[pyo3(signature = (model, inputs, path, model_name=None, trainable=false, optimizer="adam".to_string(), learning_rate=0.001))]
    pub fn trace_and_export_with_config(
        &mut self,
        model: Py<PyAny>,
        inputs: Vec<Py<PyAny>>,
        path: String,
        model_name: Option<String>,
        trainable: bool,
        optimizer: String,
        learning_rate: f32,
    ) -> PyResult<()> {
        if let Some(name) = model_name {
            self.model_name = name;
        }

        self.start_tracing();

        let result = Python::with_gil(|py| {
            let model = model.bind(py);
            let args = PyTuple::new(py, inputs.iter().map(|x| x.bind(py)))?;
            let _outputs = model.call_method("forward", args, None)?;
            self.stop_tracing()?;
            Ok::<_, PyErr>(())
        });

        if let Err(e) = result {
            self.is_tracing = false;
            return Err(e);
        }

        let cfg = self.cfg.lock().unwrap().clone();
        let dag = FullOptimizer::optimize_full(&mut cfg.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let model_file = if trainable {
            ModelFile::new_trainable(
                &self.model_name,
                &self.framework,
                dag,
                &optimizer,
                learning_rate,
            )
        } else {
            ModelFile::new(&self.model_name, &self.framework, dag)
        };

        model_file
            .export(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(())
    }

    pub fn get_cfg(&self) -> PyResult<Py<PyAny>> {
        let cfg = self.cfg.lock().unwrap();
        Python::with_gil(|py| {
            let py_cfg = PyCfgGraph { inner: cfg.clone() };
            Ok(Py::new(py, py_cfg)?.into_any())
        })
    }

    pub fn stats(&self) -> PyResult<HashMap<String, usize>> {
        let cfg = self.cfg.lock().unwrap();
        let mut stats = HashMap::new();
        stats.insert("blocks".to_string(), cfg.blocks.len());
        stats.insert(
            "ops".to_string(),
            cfg.blocks.values().map(|b| b.ops.len()).sum(),
        );
        stats.insert("values".to_string(), cfg.value_types.len());
        stats.insert("inputs".to_string(), cfg.inputs.len());
        stats.insert("outputs".to_string(), cfg.outputs.len());
        Ok(stats)
    }
}

// ============================================================
// PyCfgGraph
// ============================================================

#[pyclass]
pub struct PyCfgGraph {
    pub inner: CfgGraph,
}

#[pymethods]
impl PyCfgGraph {
    pub fn num_blocks(&self) -> usize {
        self.inner.blocks.len()
    }

    pub fn num_ops(&self) -> usize {
        self.inner.blocks.values().map(|b| b.ops.len()).sum()
    }

    pub fn num_values(&self) -> usize {
        self.inner.value_types.len()
    }

    pub fn to_dag(&self) -> PyResult<Py<PyAny>> {
        use crate::transform::FullOptimizer;

        let mut cfg = self.inner.clone();
        let dag = FullOptimizer::optimize_full(&mut cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Python::with_gil(|py| {
            let py_dag = crate::ir::serialize::PyDagGraph { inner: dag };
            Ok(Py::new(py, py_dag)?.into_any())
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyCfgGraph(blocks={}, ops={}, values={})",
            self.num_blocks(),
            self.num_ops(),
            self.num_values()
        )
    }
}

// ============================================================
// PyDagGraph
// ============================================================

#[pyclass]
pub struct PyDagGraph {
    pub inner: DagGraph,
}

#[pymethods]
impl PyDagGraph {
    pub fn num_ops(&self) -> usize {
        self.inner.ops.len()
    }

    pub fn num_values(&self) -> usize {
        self.inner.values.len()
    }

    pub fn num_constants(&self) -> usize {
        self.inner.constants.len()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DagGraph(name={}, ops={}, values={}, constants={})",
            self.inner.name,
            self.inner.ops.len(),
            self.inner.values.len(),
            self.inner.constants.len()
        )
    }
}
