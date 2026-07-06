// src/executor/executor.rs

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::ir::dag::{DagGraph, DataType};
use crate::ffi::Tensor;
use crate::pytensor::PyTensor;

use super::math;
use super::nn;
use super::activation;
use super::tensor;
use super::index;
use super::control;
use super::quantized;
use super::memory_reuse::MemoryPool;
use super::parallel::ParallelExecutor;

// ============================================================
// 执行器（带内存复用和并行）
// ============================================================
pub struct Executor {
    graph: DagGraph,
    values: HashMap<u64, Tensor>,
    memory_pool: MemoryPool,
    parallel: bool,
    num_threads: usize,
}

impl Executor {
    pub fn new(graph: DagGraph) -> Self {
        Executor {
            graph,
            values: HashMap::new(),
            memory_pool: MemoryPool::new(),
            parallel: false,
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn execute(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.values.clear();
        self.memory_pool.reset();

        // 1. 验证输入
        self.validate_inputs(inputs)?;

        // 2. 加载输入
        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            self.values.insert(input_id, inputs[i].clone());
        }

        // 3. 加载常量
        self.load_constants()?;

        // 4. 获取执行顺序
        let order = self.graph.topological_sort()?;

        // 5. 执行（串行或并行）
        if self.parallel && order.len() > 1 {
            self.execute_parallel(&order)?;
        } else {
            self.execute_serial(&order)?;
        }

        // 6. 收集输出
        self.collect_outputs()
    }

    fn validate_inputs(&self, inputs: &[Tensor]) -> Result<(), String> {
        if inputs.len() != self.graph.inputs.len() {
            return Err(format!(
                "Expected {} inputs, got {}",
                self.graph.inputs.len(),
                inputs.len()
            ));
        }

        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            if let Some(value) = self.graph.values.get(&input_id) {
                let expected_shape: Vec<usize> = value.ty.shape.iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();
                let actual_shape = inputs[i].shape();

                // 检查 shape（忽略 -1 维度）
                for (j, (&expected, &actual)) in expected_shape.iter().zip(actual_shape.iter()).enumerate() {
                    if expected != 0 && expected != actual {
                        return Err(format!(
                            "Input {} shape mismatch at dim {}: expected {}, got {}",
                            i, j, expected, actual
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn load_constants(&mut self) -> Result<(), String> {
        for (&id, data) in &self.graph.constants {
            if let Some(value) = self.graph.values.get(&id) {
                let dtype = value.ty.dtype;
                let shape: Vec<usize> = value.ty.shape.iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();

                let tensor = Self::bytes_to_tensor(data, dtype, &shape, value.scale, value.zero_point)?;
                self.values.insert(id, tensor);
            }
        }
        Ok(())
    }

    fn bytes_to_tensor(
        data: &[u8],
        dtype: DataType,
        shape: &[usize],
        scale: Option<f32>,
        zero_point: Option<f32>,
    ) -> Result<Tensor, String> {
        match dtype {
            DataType::F32 => {
                let float_data: Vec<f32> = data.chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(Tensor::new_f32(&float_data, shape))
            }
            DataType::F64 => {
                let double_data: Vec<f64> = data.chunks(8)
                    .map(|chunk| f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7]
                    ]))
                    .collect();
                Ok(Tensor::new_f64(&double_data, shape))
            }
            DataType::F16 => {
                let u16_data: Vec<u16> = data.chunks(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Ok(Tensor::new_f16(&u16_data, shape))
            }
            DataType::BF16 => {
                let u16_data: Vec<u16> = data.chunks(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Ok(Tensor::new_bf16(&u16_data, shape))
            }
            DataType::I8 => {
                let i8_data: Vec<i8> = data.iter().map(|&b| b as i8).collect();
                let scale_val = scale.unwrap_or(1.0);
                let zero_point_val = zero_point.unwrap_or(0.0);
                Ok(Tensor::new_quantized(&i8_data, shape, scale_val, zero_point_val))
            }
            DataType::I32 => {
                // 将 i32 数据转为 f32（保持数值）
                let float_data: Vec<f32> = data.chunks(4)
                    .map(|chunk| {
                        let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        val as f32
                    })
                    .collect();
                Ok(Tensor::new_f32(&float_data, shape))
            }
            DataType::I64 => {
                let float_data: Vec<f32> = data.chunks(8)
                    .map(|chunk| {
                        let val = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7]
                        ]);
                        val as f32
                    })
                    .collect();
                Ok(Tensor::new_f32(&float_data, shape))
            }
            DataType::Bool => {
                let float_data: Vec<f32> = data.iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                Ok(Tensor::new_f32(&float_data, shape))
            }
        }
    }

    fn execute_serial(&mut self, order: &[u64]) -> Result<(), String> {
        for &op_id in order {
            let op = self.graph.get_op(op_id)
                .ok_or_else(|| format!("Op {} not found", op_id))?
                .clone();
            self.execute_op(&op)?;
        }
        Ok(())
    }

    fn execute_parallel(&mut self, order: &[u64]) -> Result<(), String> {
        // 使用 ParallelExecutor
        let mut parallel_exec = ParallelExecutor::new(
            &self.graph,
            &mut self.values,
            &mut self.memory_pool,
            self.num_threads,
        );
        parallel_exec.execute(order)
    }

    fn execute_op(&mut self, op: &crate::ir::dag::Op) -> Result<(), String> {
        // 收集所有输入
        let mut input_tensors = Vec::new();
        for &in_id in &op.inputs {
            if let Some(t) = self.values.get(&in_id) {
                input_tensors.push(t.clone());
            } else {
                return Err(format!("Input value {} not found for op {}", in_id, op.id));
            }
        }

        // 验证输入 shape（使用 shape inference 结果）
        self.validate_op_inputs(op, &input_tensors)?;

        // 执行算子
        let outputs = self.dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        // 存储输出
        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                // 尝试从 memory pool 复用内存
                let tensor = self.memory_pool.allocate_or_use(outputs[i].clone());
                self.values.insert(out_id, tensor);
            }
        }

        // 标记不再需要的 tensor 为可复用
        self.mark_tensors_for_reuse(op);

        Ok(())
    }

    fn validate_op_inputs(
        &self,
        op: &crate::ir::dag::Op,
        inputs: &[Tensor],
    ) -> Result<(), String> {
        // 从 graph 中获取预期的输入 shape
        for (i, &in_id) in op.inputs.iter().enumerate() {
            if let Some(value) = self.graph.values.get(&in_id) {
                let expected_shape: Vec<usize> = value.ty.shape.iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();
                let actual_shape = inputs.get(i).map(|t| t.shape()).unwrap_or_default();

                for (j, (&expected, &actual)) in expected_shape.iter().zip(actual_shape.iter()).enumerate() {
                    if expected != 0 && expected != actual {
                        return Err(format!(
                            "Op {} input {} shape mismatch at dim {}: expected {}, got {}",
                            op.op_type, i, j, expected, actual
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn mark_tensors_for_reuse(&mut self, op: &crate::ir::dag::Op) {
        // 找出所有只被这个 op 使用的输入值
        for &in_id in &op.inputs {
            let users = self.graph.get_users(in_id);
            // 如果这个值只被当前 op 使用，且不是输出，可以复用
            if users.len() == 1 && users[0] == op.id {
                // 检查这个值是否会被后续使用
                if let Some(tensor) = self.values.get(&in_id) {
                    self.memory_pool.mark_reusable(tensor.clone());
                }
            }
        }
    }

    fn dispatch_op(
        &self,
        op_type: &str,
        inputs: &[Tensor],
        attrs: &HashMap<String, crate::ir::dag::AttrValue>,
    ) -> Result<Vec<Tensor>, String> {
        // 量化算子
        if op_type.starts_with("quantized_") {
            return quantized::dispatch_quantized(op_type, inputs, attrs);
        }

        // 按类别分派
        let dispatchers: Vec<(&str, fn(&str, &[Tensor], &HashMap<String, crate::ir::dag::AttrValue>) -> Result<Vec<Tensor>, String>)> = vec![
            ("math", math::dispatch_math),
            ("nn", nn::dispatch_nn),
            ("activation", activation::dispatch_activation),
            ("tensor", tensor::dispatch_tensor),
            ("index", index::dispatch_index),
            ("control", control::dispatch_control),
        ];

        for (_name, dispatcher) in dispatchers {
            if let Ok(result) = dispatcher(op_type, inputs, attrs) {
                return Ok(result);
            }
        }

        Err(format!("Unknown operator: {}", op_type))
    }

    fn collect_outputs(&self) -> Result<Vec<Tensor>, String> {
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
    #[new]
    pub fn new(graph: Py<PyAny>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let graph_obj = graph.bind(py);
            if let Ok(py_dag) = graph_obj.downcast::<crate::ir::serialize::PyDagGraph>() {
                let dag = py_dag.borrow().inner.clone();
                Ok(PyExecutor {
                    inner: Executor::new(dag),
                })
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected PyDagGraph"
                ))
            }
        })
    }

    #[staticmethod]
    pub fn from_model_file(model_file: &crate::ir::serialize::PyModelFile) -> Self {
        let graph = model_file.inner.graph.clone();
        PyExecutor {
            inner: Executor::new(graph),
        }
    }

    // ✅ 修复：参数类型改为 Py<PyList>，手动提取
    pub fn execute(&mut self, inputs: Py<PyList>) -> PyResult<Vec<PyTensor>> {
        Python::with_gil(|py| {
            let inputs_list = inputs.bind(py);

            let mut input_tensors = Vec::new();
            for item in inputs_list.iter() {
                let pytensor = item.downcast::<PyTensor>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected PyTensor"))?;
                input_tensors.push(pytensor.borrow().inner.clone());
            }

            let result = self.inner.execute(&input_tensors)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            Ok(result.into_iter().map(|t| PyTensor { inner: t }).collect())
        })
    }

    pub fn set_parallel(&mut self, parallel: bool) {
        self.inner.parallel = parallel;
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.inner.num_threads = threads;
    }

    pub fn memory_stats(&self) -> PyResult<HashMap<String, usize>> {
        let (allocations, reuse_count) = self.inner.memory_pool.stats();
        let mut stats = HashMap::new();
        stats.insert("allocations".to_string(), allocations);
        stats.insert("reuses".to_string(), reuse_count);
        Ok(stats)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyExecutor(ops={}, parallel={})",
            self.inner.graph.ops.len(),
            self.inner.parallel
        )
    }
}


/// 导出 dispatch_op 供其他模块使用
pub use super::parallel::dispatch_op;
