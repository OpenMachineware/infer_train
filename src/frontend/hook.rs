// src/frontend/hook.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ir::cfg::{CfgGraph, CfgOp, BranchInfo};
use crate::ir::dag::{AttrValue, DataType, TensorType};
use crate::ir::dag::DagGraph;
use crate::transform::FullOptimizer;
use crate::ir::serialize::ModelFile;

/// Hook 追踪器 - 捕获 PyTorch 模型执行
#[pyclass]
pub struct HookTracer {
    cfg: Arc<Mutex<CfgGraph>>,
    block_stack: Vec<u64>,
    current_block: u64,
    value_map: HashMap<u64, u64>,
    op_counter: u64,
    is_tracing: bool,
    block_counter: u64,
    // 记录模型信息
    model_name: String,
    framework: String,
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
        }
    }

    // ============================================================
    // 基础追踪 API
    // ============================================================

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
    }

    pub fn set_model_name(&mut self, name: String) {
        self.model_name = name;
    }

    pub fn set_framework(&mut self, framework: String) {
        self.framework = framework;
    }

    // ============================================================
    // 记录算子（由 Python hook 调用）
    // ============================================================

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

        let input_ids: Result<Vec<u64>, PyErr> = inputs.iter()
            .map(|obj| self.extract_value_id(obj))
            .collect();
        let input_ids = input_ids?;

        let output_ids: Result<Vec<u64>, PyErr> = outputs.iter()
            .map(|obj| self.extract_value_id(obj))
            .collect();
        let output_ids = output_ids?;

        let rust_attrs = self.convert_attrs(&attrs)?;

        let op_id = self.op_counter;
        self.op_counter += 1;

        let op_name = name.unwrap_or_else(|| format!("{}_{}", op_type, op_id));

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
            cfg.add_op(self.current_block, cfg_op)?;
        }

        // 更新 value 映射和类型信息
        for (i, out_id) in output_ids.iter().enumerate() {
            self.value_map.insert(*out_id, *out_id);
            if i < outputs.len() {
                if let Ok((dtype, shape)) = self.extract_tensor_info(&outputs[i]) {
                    let mut cfg = self.cfg.lock().unwrap();
                    cfg.value_types.insert(*out_id, (dtype, shape));
                }
            }
        }

        Ok(())
    }

    // ============================================================
    // 控制流记录
    // ============================================================

    pub fn begin_block(&mut self, name: Option<String>) -> PyResult<u64> {
        if !self.is_tracing {
            return Ok(0);
        }

        let block_name = name.unwrap_or_else(|| format!("block_{}", self.block_counter));
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

        let cond_id = self.extract_value_id(&condition)?;

        let merge_name = format!("merge_{}", self.block_counter);
        self.block_counter += 1;

        let mut cfg = self.cfg.lock().unwrap();
        let merge_id = cfg.add_block(&merge_name);

        cfg.set_branch(self.current_block, cond_id, true_block, false_block, merge_id)?;
        cfg.add_edge(self.current_block, true_block)?;
        cfg.add_edge(self.current_block, false_block)?;
        cfg.add_edge(true_block, merge_id)?;
        cfg.add_edge(false_block, merge_id)?;

        self.current_block = merge_id;
        self.block_stack.push(merge_id);

        Ok(())
    }

    // ============================================================
    // 高层 API：用户直接调用
    // ============================================================

    /// 追踪模型并返回优化后的 DAG
    pub fn trace_to_dag(
        &mut self,
        model: Py<PyAny>,
        inputs: Vec<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // 1. 追踪
        self.start_tracing();

        let result = Python::with_gil(|py| {
            let model = model.as_ref(py);
            let inputs: Vec<&PyAny> = inputs.iter().map(|x| x.as_ref(py)).collect();

            let outputs = model.call_method("forward", &inputs, None)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            self.stop_tracing()?;
            Ok::<_, PyErr>(outputs)
        });

        if let Err(e) = result {
            self.is_tracing = false;
            return Err(e);
        }

        // 2. CFG → DAG（含优化）
        let cfg = self.cfg.lock().unwrap().clone();
        let dag = FullOptimizer::optimize_full(&mut cfg.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Python::with_gil(|py| {
            let py_dag = crate::ir::serialize::PyDagGraph { inner: dag };
            Ok(Py::new(py, py_dag)?)
        })
    }

    /// 追踪、优化并导出为 ITM 文件（推理模式）
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
            false,  // trainable
            "adam".to_string(),
            0.001,
        )
    }

    /// 追踪、优化并导出为 ITM 文件（可训练模式）
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
            true,  // trainable
            optimizer,
            learning_rate,
        )
    }

    /// 完整配置导出
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
        // 1. 更新名称
        if let Some(name) = model_name {
            self.model_name = name;
        }

        // 2. 追踪
        self.start_tracing();

        let result = Python::with_gil(|py| {
            let model = model.as_ref(py);
            let inputs: Vec<&PyAny> = inputs.iter().map(|x| x.as_ref(py)).collect();

            let _outputs = model.call_method("forward", &inputs, None)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            self.stop_tracing()?;
            Ok::<_, PyErr>(())
        });

        if let Err(e) = result {
            self.is_tracing = false;
            return Err(e);
        }

        // 3. 获取 CFG 并优化
        let cfg = self.cfg.lock().unwrap().clone();
        let dag = FullOptimizer::optimize_full(&mut cfg.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // 4. 创建 ModelFile
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

        // 5. 导出
        model_file.export(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        println!("[HookTracer] Model exported to: {}", path);
        if trainable {
            println!("[HookTracer] Trainable mode: optimizer={}, lr={}", optimizer, learning_rate);
        }

        Ok(())
    }

    /// 获取 CFG（用于调试）
    pub fn get_cfg(&self) -> PyResult<Py<PyAny>> {
        let cfg = self.cfg.lock().unwrap();
        Python::with_gil(|py| {
            let py_cfg = PyCfgGraph { inner: cfg.clone() };
            Ok(Py::new(py, py_cfg)?)
        })
    }

    /// 获取统计信息
    pub fn stats(&self) -> PyResult<HashMap<String, usize>> {
        let cfg = self.cfg.lock().unwrap();
        let mut stats = HashMap::new();
        stats.insert("blocks".to_string(), cfg.blocks.len());
        stats.insert("ops".to_string(), cfg.blocks.values().map(|b| b.ops.len()).sum());
        stats.insert("values".to_string(), cfg.value_types.len());
        stats.insert("inputs".to_string(), cfg.inputs.len());
        stats.insert("outputs".to_string(), cfg.outputs.len());
        Ok(stats)
    }

    // ============================================================
    // 辅助方法（私有）
    // ============================================================

    fn extract_value_id(&self, obj: &Py<PyAny>) -> PyResult<u64> {
        Python::with_gil(|py| {
            let obj = obj.as_ref(py);

            // 如果是 PyTensor
            if let Ok(pytensor) = obj.downcast::<crate::pytensor::PyTensor>() {
                let tensor = pytensor.borrow().inner.as_ref();
                return Ok(tensor as *const _ as u64);
            }

            // 如果是 torch.Tensor，用 id()
            if let Ok(py_id) = obj.call_method0("__hash__") {
                if let Ok(id_val) = py_id.extract::<u64>() {
                    return Ok(id_val);
                }
            }

            // Fallback: 对象地址
            Ok(obj as *const _ as u64)
        })
    }

    fn extract_tensor_info(&self, obj: &Py<PyAny>) -> PyResult<(DataType, Vec<i64>)> {
        Python::with_gil(|py| {
            let obj = obj.as_ref(py);

            // 如果是 PyTensor
            if let Ok(pytensor) = obj.downcast::<crate::pytensor::PyTensor>() {
                let tensor = pytensor.borrow().inner.as_ref();
                let dtype = tensor.dtype();
                let shape = tensor.shape();

                let rust_dtype = match dtype.as_str() {
                    "f32" => DataType::F32,
                    "f64" => DataType::F64,
                    "f16" => DataType::F16,
                    "bf16" => DataType::BF16,
                    "i8" => DataType::I8,
                    "i32" => DataType::I32,
                    "i64" => DataType::I64,
                    "bool" => DataType::Bool,
                    _ => DataType::F32,
                };

                let rust_shape: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
                return Ok((rust_dtype, rust_shape));
            }

            // 如果是 torch.Tensor
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
                        s if s.contains("float32") || s.contains("f32") => DataType::F32,
                        s if s.contains("float64") || s.contains("f64") => DataType::F64,
                        s if s.contains("float16") || s.contains("f16") => DataType::F16,
                        s if s.contains("bfloat16") || s.contains("bf16") => DataType::BF16,
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
        })
    }

    fn convert_attrs(&self, attrs: &HashMap<String, PyObject>) -> PyResult<HashMap<String, AttrValue>> {
        let mut result = HashMap::new();

        Python::with_gil(|py| {
            for (key, value) in attrs {
                let val = value.as_ref(py);

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
            Ok(())
        })
    }
}

// ============================================================
// CFG 的 Python 包装
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
            Ok(Py::new(py, py_dag)?)
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
// DAG 的 Python 包装
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

    pub fn __repr__(&self) -> String {
        format!("DagGraph(name={}, ops={}, values={})",
                self.inner.name,
                self.inner.ops.len(),
                self.inner.values.len()
        )
    }
}
