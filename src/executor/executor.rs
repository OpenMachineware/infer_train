// src/executor/executor.rs

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;

use super::scheduler::{Scheduler, SchedulerConfig};
use crate::ir::dag::DagGraph;
use crate::tensor::Tensor;

use super::activation;
use super::control;
use super::index;
use super::math;
use super::memory_reuse::{MemoryConfig, MemoryPool};
use super::nn;
use super::parallel::ParallelExecutor;
use super::quantized;
use super::tensor;

// ============================================================
// Executor
// ============================================================
pub struct Executor {
    graph: DagGraph,
    values: HashMap<u64, Tensor<f32>>,
    memory_pool: MemoryPool,
    scheduler: Scheduler,
    parallel: bool,
    num_threads: usize,
}

impl Executor {
    pub fn new(graph: DagGraph) -> Self {
        Self::with_config(
            graph,
            MemoryConfig::inference(),
            SchedulerConfig::default(),
        )
    }

    pub fn with_memory_config(graph: DagGraph, config: MemoryConfig) -> Self {
        Self::with_config(graph, config, SchedulerConfig::default())
    }

    pub fn with_scheduler_config(
        graph: DagGraph,
        config: SchedulerConfig,
    ) -> Self {
        Self::with_config(graph, MemoryConfig::inference(), config)
    }

    pub fn with_config(
        graph: DagGraph,
        memory_config: MemoryConfig,
        scheduler_config: SchedulerConfig,
    ) -> Self {
        let mut scheduler = Scheduler::new(scheduler_config);
        scheduler.schedule(&graph);

        Executor {
            graph,
            values: HashMap::new(),
            memory_pool: MemoryPool::new(
                memory_config.block_size,
                memory_config.total_size,
            ),
            scheduler,
            parallel: false,
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn enable_training_mode(&mut self) {
        let config = MemoryConfig::training();
        self.memory_pool =
            MemoryPool::new(config.block_size, config.total_size);
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn execute(
        &mut self,
        inputs: &[Tensor<f32>],
    ) -> Result<Vec<Tensor<f32>>, String> {
        self.values.clear();
        self.memory_pool.reset();

        self.validate_inputs(inputs)?;

        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            self.values.insert(input_id, inputs[i].clone());
        }

        self.load_constants()?;

        let order = self.scheduler.get_execution_order().to_vec();

        if self.parallel && order.len() > 1 {
            self.execute_parallel(&order)?;
        } else {
            self.execute_serial(&order)?;
        }

        self.collect_outputs()
    }

    pub fn execute_batch(
        &mut self,
        batch_inputs: &[Vec<Tensor<f32>>],
    ) -> Result<Vec<Vec<Tensor<f32>>>, String> {
        let mut results = Vec::new();

        for inputs in batch_inputs {
            let result = self.execute(inputs)?;
            results.push(result);
        }

        Ok(results)
    }

    pub fn execute_batch_single(
        &mut self,
        inputs: &[Tensor<f32>],
        batch_size: usize,
    ) -> Result<Vec<Vec<Tensor<f32>>>, String> {
        // Split inputs into chunks by batch_size
        let mut results = Vec::new();
        let total = inputs.len();

        for start in (0..total).step_by(batch_size) {
            let end = (start + batch_size).min(total);
            let batch = &inputs[start..end];
            let result = self.execute(batch)?;
            results.push(result);
        }

        Ok(results)
    }

    pub fn execute_batch_multi(
        &mut self,
        input_batches: &[Vec<Tensor<f32>>],
    ) -> Result<Vec<Vec<Tensor<f32>>>, String> {
        let mut results = Vec::new();

        for batch in input_batches {
            let result = self.execute(batch)?;
            results.push(result);
        }

        Ok(results)
    }

    // Optimized batch inference
    // (share constant loading, execute multiple batches at once)
    pub fn execute_batch_optimized(
        &mut self,
        batches: &[Vec<Tensor<f32>>],
    ) -> Result<Vec<Vec<Tensor<f32>>>, String> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }

        // Load constants (only once)
        self.load_constants()?;

        // Get execution order (only once)
        let order = self.scheduler.get_execution_order().to_vec();

        // Execute for each batch
        let mut all_results = Vec::with_capacity(batches.len());

        for inputs in batches {
            // Clear previous intermediate values, but keep constants
            self.values.clear();
            self.memory_pool.reset();

            // Only load inputs
            for (i, &input_id) in self.graph.inputs.iter().enumerate() {
                if i < inputs.len() {
                    self.values.insert(input_id, inputs[i].clone());
                }
            }

            // Execute
            if self.parallel && order.len() > 1 {
                self.execute_parallel(&order)?;
            } else {
                self.execute_serial(&order)?;
            }

            all_results.push(self.collect_outputs()?);
        }

        Ok(all_results)
    }

    fn validate_inputs(&self, inputs: &[Tensor<f32>]) -> Result<(), String> {
        if inputs.len() != self.graph.inputs.len() {
            return Err(format!(
                "Expected {} inputs, got {}",
                self.graph.inputs.len(),
                inputs.len()
            ));
        }
        Ok(())
    }

    fn load_constants(&mut self) -> Result<(), String> {
        for (&id, data) in &self.graph.constants {
            if let Some(value) = self.graph.values.get(&id) {
                let shape: Vec<usize> = value
                    .ty
                    .shape
                    .iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();

                let tensor = Self::bytes_to_tensor(data, &shape)?;
                self.values.insert(id, tensor);
            }
        }
        Ok(())
    }

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

    fn execute_serial(&mut self, order: &[u64]) -> Result<(), String> {
        for &op_id in order {
            let op = self
                .graph
                .get_op(op_id)
                .ok_or_else(|| format!("Op {} not found", op_id))?
                .clone();
            self.execute_op(&op)?;
        }
        Ok(())
    }

    fn execute_parallel(&mut self, order: &[u64]) -> Result<(), String> {
        let mut parallel_exec = ParallelExecutor::new(
            &self.graph,
            &mut self.values,
            &mut self.memory_pool,
            self.num_threads,
        );
        parallel_exec.execute(order)
    }

    fn execute_op(&mut self, op: &crate::ir::dag::Op) -> Result<(), String> {
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

        let outputs =
            self.dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                // Get bytes first, then allocate
                let data_bytes = outputs[i].data();
                let bytes = bytemuck::cast_slice(data_bytes);

                if let Some(id) = self.memory_pool.allocate(bytes) {
                    // Get data from memory pool and create Tensor
                    if let Some(pool_data) = self.memory_pool.get_mut(id) {
                        pool_data[..bytes.len()].copy_from_slice(bytes);
                        let float_data: &[f32] =
                            bytemuck::cast_slice(pool_data);
                        let tensor = Tensor::new(
                            float_data.to_vec(),
                            outputs[i].shape(),
                        );
                        self.values.insert(out_id, tensor);
                    } else {
                        // Fallback: store directly
                        self.values.insert(out_id, outputs[i].clone());
                    }
                } else {
                    // Memory pool full, store directly
                    self.values.insert(out_id, outputs[i].clone());
                }
            }
        }

        self.mark_tensors_for_reuse(op);

        Ok(())
    }

    fn mark_tensors_for_reuse(&mut self, op: &crate::ir::dag::Op) {
        for &in_id in &op.inputs {
            let users = self.graph.get_users(in_id);
            if users.len() == 1 && users[0] == op.id {
                if let Some(_tensor) = self.values.get(&in_id) {
                    self.memory_pool.mark_reusable(in_id);
                }
            }
        }
    }

    fn dispatch_op(
        &self,
        op_type: &str,
        inputs: &[Tensor<f32>],
        attrs: &HashMap<String, crate::ir::dag::AttrValue>,
    ) -> Result<Vec<Tensor<f32>>, String> {
        if op_type.starts_with("quantized_") {
            return quantized::dispatch_quantized(op_type, inputs, attrs);
        }

        let dispatchers: Vec<(
            &str,
            fn(
                &str,
                &[Tensor<f32>],
                &HashMap<String, crate::ir::dag::AttrValue>,
            ) -> Result<Vec<Tensor<f32>>, String>,
        )> = vec![
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

    fn collect_outputs(&self) -> Result<Vec<Tensor<f32>>, String> {
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
// dispatch_op function (exported for parallel use)
// ============================================================

pub fn dispatch_op(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, crate::ir::dag::AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    if op_type.starts_with("quantized_") {
        return quantized::dispatch_quantized(op_type, inputs, attrs);
    }

    let dispatchers: Vec<(
        &str,
        fn(
            &str,
            &[Tensor<f32>],
            &HashMap<String, crate::ir::dag::AttrValue>,
        ) -> Result<Vec<Tensor<f32>>, String>,
    )> = vec![
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

// ============================================================
// PyO3 Bindings
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
            if let Ok(py_dag) =
                graph_obj.downcast::<crate::ir::serialize::PyDagGraph>()
            {
                let dag = py_dag.borrow().inner.clone();
                Ok(PyExecutor { inner: Executor::new(dag) })
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected PyDagGraph",
                ))
            }
        })
    }

    #[staticmethod]
    pub fn from_model_file(
        model_file: &crate::ir::serialize::PyModelFile,
    ) -> Self {
        let guard = model_file.inner.lock().unwrap();
        let graph = guard.graph().clone();
        PyExecutor { inner: Executor::new(graph) }
    }

    pub fn execute(&mut self, inputs: Py<PyList>) -> PyResult<Vec<Py<PyAny>>> {
        Python::with_gil(|py| {
            let inputs_list = inputs.bind(py);
            let mut input_tensors = Vec::new();
            for item in inputs_list.iter() {
                // Extract data from PyObject
                let data: Vec<f32> = item.extract()?;
                let shape = vec![data.len()];
                input_tensors.push(Tensor::new(data, &shape));
            }

            let result = self
                .inner
                .execute(&input_tensors)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            let py_result: Vec<Py<PyAny>> = result
                .into_iter()
                .map(|t| {
                    let data = t.data().to_vec();
                    PyList::new(py, data).unwrap().into_any().unbind()
                })
                .collect();
            Ok(py_result)
        })
    }

    // Basic batch inference
    pub fn execute_batch(
        &mut self,
        inputs: Vec<Vec<Py<PyAny>>>,
    ) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        Python::with_gil(|py| {
            let mut batch_tensors = Vec::new();
            for batch in inputs {
                let mut input_tensors = Vec::new();
                for item in batch {
                    let data: Vec<f32> = item.extract(py)?;
                    let shape = vec![data.len()];
                    input_tensors.push(Tensor::new(data, &shape));
                }
                batch_tensors.push(input_tensors);
            }

            let results = self
                .inner
                .execute_batch(&batch_tensors)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            let py_results: Vec<Vec<Py<PyAny>>> = results
                .into_iter()
                .map(|batch| {
                    batch
                        .into_iter()
                        .map(|t| {
                            let data = t.data().to_vec();
                            PyList::new(py, data).unwrap().into_any().unbind()
                        })
                        .collect()
                })
                .collect();

            Ok(py_results)
        })
    }

    // Optimized batch inference (shared constants)
    pub fn execute_batch_optimized(
        &mut self,
        inputs: Vec<Vec<Py<PyAny>>>,
    ) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        Python::with_gil(|py| {
            let mut batch_tensors = Vec::new();
            for batch in inputs {
                let mut input_tensors = Vec::new();
                for item in batch {
                    let data: Vec<f32> = item.extract(py)?;
                    let shape = vec![data.len()];
                    input_tensors.push(Tensor::new(data, &shape));
                }
                batch_tensors.push(input_tensors);
            }

            let results = self
                .inner
                .execute_batch_optimized(&batch_tensors)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            let py_results: Vec<Vec<Py<PyAny>>> = results
                .into_iter()
                .map(|batch| {
                    batch
                        .into_iter()
                        .map(|t| {
                            let data = t.data().to_vec();
                            PyList::new(py, data).unwrap().into_any().unbind()
                        })
                        .collect()
                })
                .collect();

            Ok(py_results)
        })
    }

    pub fn set_parallel(&mut self, parallel: bool) {
        self.inner.parallel = parallel;
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.inner.num_threads = threads;
    }

    pub fn memory_stats(&self) -> PyResult<HashMap<String, usize>> {
        let (allocations, reuse_count, allocated_size, used_size) =
            self.inner.memory_pool.stats();
        let mut stats = HashMap::new();
        stats.insert("allocations".to_string(), allocations);
        stats.insert("reuses".to_string(), reuse_count);
        stats.insert("allocated_size".to_string(), allocated_size);
        stats.insert("used_size".to_string(), used_size);
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
