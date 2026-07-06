// src/frontend/hook.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ir::cfg::{CfgGraph, CfgOp};
use crate::ir::dag::{AttrValue, DataType, TensorType};
use crate::transform::cfg_to_dag::CfgToDagConverter;
use crate::ir::dag::DagGraph;

/// Hook 追踪器 - 捕获 PyTorch 模型执行
#[pyclass]
pub struct HookTracer {
    // CFG 图
    cfg: Arc<Mutex<CfgGraph>>,
    // 当前块栈
    block_stack: Vec<u64>,
    current_block: u64,
    // 值映射: Python object id -> CFG value id
    value_map: HashMap<u64, u64>,
    // 算子计数器
    op_counter: u64,
    // 是否在追踪
    is_tracing: bool,
    // 块名称计数器
    block_counter: u64,
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
        }
    }

    /// 开始追踪
    pub fn start_tracing(&mut self) {
        self.is_tracing = true;
        self.value_map.clear();
        self.op_counter = 0;
        self.block_counter = 1;
        println!("[HookTracer] Tracing started");
    }

    /// 停止追踪
    pub fn stop_tracing(&mut self) -> PyResult<()> {
        self.is_tracing = false;
        println!("[HookTracer] Tracing stopped");
        Ok(())
    }

    /// 记录算子调用（由 Python hook 调用）
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

        // 提取输入 IDs
        let input_ids: Result<Vec<u64>, PyErr> = inputs.iter()
            .map(|obj| self.extract_value_id(obj))
            .collect();
        let input_ids = input_ids?;

        // 提取输出 IDs
        let output_ids: Result<Vec<u64>, PyErr> = outputs.iter()
            .map(|obj| self.extract_value_id(obj))
            .collect();
        let output_ids = output_ids?;

        // 转换 attrs
        let rust_attrs = self.convert_attrs(&attrs)?;

        // 添加算子到当前块
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

        // 更新 value 映射
        for (i, out_id) in output_ids.iter().enumerate() {
            self.value_map.insert(*out_id, *out_id);

            // 提取 shape 和 dtype（从 Python 对象）
            if i < outputs.len() {
                if let Ok(info) = self.extract_tensor_info(&outputs[i]) {
                    let mut cfg = self.cfg.lock().unwrap();
                    cfg.value_types.insert(*out_id, info);
                }
            }
        }

        Ok(())
    }

    /// 开始一个新块（用于控制流）
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

    /// 结束当前块
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

    /// 记录条件分支
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

        // 创建合并块
        let merge_name = format!("merge_{}", self.block_counter);
        self.block_counter += 1;

        let mut cfg = self.cfg.lock().unwrap();
        let merge_id = cfg.add_block(&merge_name);

        // 设置分支信息
        cfg.set_branch(self.current_block, cond_id, true_block, false_block, merge_id)?;

        // 添加边
        cfg.add_edge(self.current_block, true_block)?;
        cfg.add_edge(self.current_block, false_block)?;
        cfg.add_edge(true_block, merge_id)?;
        cfg.add_edge(false_block, merge_id)?;

        // 切换到合并块
        self.current_block = merge_id;
        self.block_stack.push(merge_id);

        Ok(())
    }

    /// 获取追踪结果（CFG）
    pub fn get_cfg(&self) -> PyResult<Py<PyAny>> {
        let cfg = self.cfg.lock().unwrap();
        Python::with_gil(|py| {
            // 转换为 Python 可读的格式
            let dict = PyDict::new(py);
            dict.set_item("name", cfg.name.clone())?;

            // 返回 CFG 对象
            Ok(Py::new(py, PyCfgGraph { inner: cfg.clone() })?)
        })
    }

    /// 转换为 DAG
    pub fn into_dag(&mut self) -> PyResult<Py<PyAny>> {
        // 停止追踪
        self.is_tracing = false;

        let cfg = self.cfg.lock().unwrap();
        let dag = CfgToDagConverter::convert(&cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Python::with_gil(|py| {
            // 创建 PyDagGraph
            let py_dag = PyDagGraph { inner: dag };
            Ok(Py::new(py, py_dag)?)
        })
    }

    /// 重置追踪器
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

    /// 转换为 DAG 并自动优化
    pub fn into_optimized_dag(&mut self) -> PyResult<Py<PyAny>> {
        let dag = self.into_dag()?;

        // 获取 PyDagGraph
        Python::with_gil(|py| {
            let py_dag = dag.extract::<PyDagGraph>(py)?;

            // 优化
            let mut inner = py_dag.inner.clone();
            crate::transform::Optimizer::optimize(&mut inner)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            let optimized = PyDagGraph { inner };
            Ok(Py::new(py, optimized)?)
        })
    }

    /// 一步到位：追踪 + 转换 + 优化 + 保存
    pub fn trace_and_save(
        &mut self,
        model: Py<PyAny>,
        inputs: Vec<Py<PyAny>>,
        path: String,
        model_name: String,
        trainable: bool,
    ) -> PyResult<()> {
        // 1. 追踪
        self.start_tracing();

        Python::with_gil(|py| {
            let model = model.as_ref(py);
            let inputs: Vec<&PyAny> = inputs.iter().map(|x| x.as_ref(py)).collect();

            // 执行模型
            let _outputs = model.call_method("forward", &inputs, None)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            self.stop_tracing()?;
            Ok(())
        })?;

        // 2. 获取 CFG
        let cfg = self.cfg.lock().unwrap();

        // 3. 转换 → DAG
        let mut dag = crate::transform::cfg_to_dag::CfgToDagConverter::convert(&cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // 4. 优化
        crate::transform::Optimizer::optimize(&mut dag)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // 5. 保存
        let model_file = if trainable {
            crate::ir::serialize::ModelFile::new_trainable(
                &model_name,
                "torch",
                dag,
                "adam",
                0.001,
            )
        } else {
            crate::ir::serialize::ModelFile::new(&model_name, "torch", dag)
        };

        model_file.export(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(())
    }

    /// 追踪并完整优化（CFG + DAG）
    pub fn trace_optimize_and_export(
        &mut self,
        model: Py<PyAny>,
        inputs: Vec<Py<PyAny>>,
        path: String,
        model_name: String,
        trainable: bool,
    ) -> PyResult<()> {
        // 1. 追踪
        self.start_tracing();

        Python::with_gil(|py| {
            let model = model.as_ref(py);
            let inputs: Vec<&PyAny> = inputs.iter().map(|x| x.as_ref(py)).collect();

            let _outputs = model.call_method("forward", &inputs, None)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            self.stop_tracing()?;
            Ok(())
        })?;

        // 2. 获取 CFG
        let mut cfg = self.cfg.lock().unwrap().clone();

        // 3. CFG 优化 + DAG 转换 + DAG 优化
        let dag = crate::transform::FullOptimizer::optimize_full(&mut cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // 4. 保存
        let model_file = if trainable {
            crate::ir::serialize::ModelFile::new_trainable(
                &model_name,
                "torch",
                dag,
                "adam",
                0.001,
            )
        } else {
            crate::ir::serialize::ModelFile::new(&model_name, "torch", dag)
        };

        model_file.export(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(())
    }

    // ============================================================
    // 辅助方法
    // ============================================================

    /// 从 Python 对象提取 value ID
    fn extract_value_id(&self, obj: &Py<PyAny>) -> PyResult<u64> {
        Python::with_gil(|py| {
            let obj = obj.as_ref(py);

            // 如果是 PyTensor，使用其内部指针
            if let Ok(pytensor) = obj.downcast::<crate::pytensor::PyTensor>() {
                let tensor = pytensor.borrow().inner.as_ref();
                return Ok(tensor as *const _ as u64);
            }

            // 如果是 torch.Tensor，使用 Python id()
            if let Ok(py_id) = obj.call_method0("__hash__") {
                if let Ok(id_val) = py_id.extract::<u64>() {
                    return Ok(id_val);
                }
            }

            // Fallback: 使用对象地址
            let ptr = obj as *const _ as u64;
            Ok(ptr)
        })
    }

    /// 提取 tensor 信息（shape + dtype）
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

            // 如果是 torch.Tensor，通过 Python 获取
            if let Ok(shape) = obj.call_method0("shape") {
                if let Ok(shape_list) = shape.extract::<Vec<i64>>() {
                    // 获取 dtype
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

    /// 转换 Python attrs 到 Rust AttrValue
    fn convert_attrs(&self, attrs: &HashMap<String, PyObject>) -> PyResult<HashMap<String, AttrValue>> {
        let mut result = HashMap::new();

        Python::with_gil(|py| {
            for (key, value) in attrs {
                let val = value.as_ref(py);

                // 尝试各种类型转换
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
                } else if let Ok(tuple) = val.extract::<(i64, i64)>() {
                    result.insert(key.clone(), AttrValue::IntList(vec![tuple.0, tuple.1]));
                } else if let Ok(tuple) = val.extract::<(i64, i64, i64)>() {
                    result.insert(key.clone(), AttrValue::IntList(vec![tuple.0, tuple.1, tuple.2]));
                } else if let Ok(tuple) = val.extract::<(i64, i64, i64, i64)>() {
                    result.insert(key.clone(), AttrValue::IntList(vec![tuple.0, tuple.1, tuple.2, tuple.3]));
                } else {
                    // 跳过无法转换的属性
                    eprintln!("[HookTracer] Warning: Cannot convert attribute: {}", key);
                }
            }
            Ok(())
        })
    }
}

// ============================================================
// Python 可用的 CFG 包装
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
        self.inner.blocks.values()
            .map(|b| b.ops.len())
            .sum()
    }

    pub fn __repr__(&self) -> String {
        format!("CfgGraph(name={}, blocks={}, ops={})",
                self.inner.name,
                self.inner.blocks.len(),
                self.num_ops()
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
