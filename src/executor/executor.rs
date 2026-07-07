// src/executor/executor.rs

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyList};

use crate::ir::dag::{DagGraph, DataType};
use crate::tensor::Tensor;
use crate::ir::serialize::ModelFile;

use super::math;
use super::nn;
use super::activation;
use super::tensor;
use super::index;
use super::control;
use super::quantized;
use super::parallel::ParallelExecutor;
use super::memory_reuse::{MemoryPool, MemoryConfig};

// ============================================================
// 执行器
// ============================================================
pub struct Executor {
    graph: DagGraph,
    values: HashMap<u64, Tensor<f32>>,
    memory_pool: MemoryPool,
    parallel: bool,
    num_threads: usize,
}

impl Executor {
    pub fn new(graph: DagGraph) -> Self {
        Self::with_memory_config(graph, MemoryConfig::inference())
    }

    pub fn with_memory_config(graph: DagGraph, config: MemoryConfig) -> Self {
        Executor {
            graph,
            values: HashMap::new(),
            memory_pool: MemoryPool::new(config.block_size, config.total_size),
            parallel: false,
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn enable_training_mode(&mut self) {
        let config = MemoryConfig::training();
        self.memory_pool = MemoryPool::new(config.block_size, config.total_size);
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn execute(&mut self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>, String> {
        self.values.clear();
        self.memory_pool.reset();

        self.validate_inputs(inputs)?;

        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            self.values.insert(input_id, inputs[i].clone());
        }

        self.load_constants()?;

        let order = self.graph.topological_sort()?;

        if self.parallel && order.len() > 1 {
            self.execute_parallel(&order)?;
        } else {
            self.execute_serial(&order)?;
        }

        self.collect_outputs()
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
                let shape: Vec<usize> = value.ty.shape.iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();

                let tensor = Self::bytes_to_tensor(data, &shape)?;
                self.values.insert(id, tensor);
            }
        }
        Ok(())
    }

    fn bytes_to_tensor(data: &[u8], shape: &[usize]) -> Result<Tensor<f32>, String> {
        let float_data: Vec<f32> = data.chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(Tensor::new(float_data, shape))
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
                return Err(format!("Input value {} not found for op {}", in_id, op.id));
            }
        }

        let outputs = self.dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                let tensor = self.memory_pool.allocate_or_use(outputs[i].clone());
                self.values.insert(out_id, tensor);
            }
        }

        self.mark_tensors_for_reuse(op);

        Ok(())
    }

    fn mark_tensors_for_reuse(&mut self, op: &crate::ir::dag::Op) {
        for &in_id in &op.inputs {
            let users = self.graph.get_users(in_id);
            if users.len() == 1 && users[0] == op.id {
                if let Some(tensor) = self.values.get(&in_id) {
                    self.memory_pool.mark_reusable(tensor);
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

        let dispatchers: Vec<(&str, fn(&str, &[Tensor<f32>], &HashMap<String, crate::ir::dag::AttrValue>) -> Result<Vec<Tensor<f32>>, String>)> = vec![
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
// dispatch_op 函数（导出供 parallel 使用）
// ============================================================

pub fn dispatch_op(
    op_type: &str,
    inputs: &[Tensor<f32>],
    attrs: &HashMap<String, crate::ir::dag::AttrValue>,
) -> Result<Vec<Tensor<f32>>, String> {
    if op_type.starts_with("quantized_") {
        return quantized::dispatch_quantized(op_type, inputs, attrs);
    }

    let dispatchers: Vec<(&str, fn(&str, &[Tensor<f32>], &HashMap<String, crate::ir::dag::AttrValue>) -> Result<Vec<Tensor<f32>>, String>)> = vec![
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
// PyO3 绑定
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
        let guard = model_file.inner.lock().unwrap();
        let graph = guard.graph().clone();
        PyExecutor {
            inner: Executor::new(graph),
        }
    }

    pub fn execute(&mut self, inputs: Py<PyList>) -> PyResult<Vec<Py<PyAny>>> {
        Python::with_gil(|py| {
            let inputs_list = inputs.bind(py);
            let mut input_tensors = Vec::new();
            for item in inputs_list.iter() {
                // 从 PyObject 提取数据
                let data: Vec<f32> = item.extract()?;
                let shape = vec![data.len()];
                input_tensors.push(Tensor::new(data, &shape));
            }

            let result = self.inner.execute(&input_tensors)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            let py_result: Vec<Py<PyAny>> = result.into_iter()
                .map(|t| {
                    let data = t.data().to_vec();
                    PyList::new(py, data).unwrap().into_any().unbind()
                })
                .collect();
            Ok(py_result)
        })
    }

    pub fn set_parallel(&mut self, parallel: bool) {
        self.inner.parallel = parallel;
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.inner.num_threads = threads;
    }

    pub fn memory_stats(&self) -> PyResult<HashMap<String, usize>> {
        let (allocations, reuse_count, allocated_size, used_size) = self.inner.memory_pool.stats();
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

// ============================================================
// 训练器配置
// ============================================================
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub learning_rate: f32,
    pub optimizer_type: OptimizerType,
    pub weight_decay: f32,
    pub gradient_clip: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            learning_rate: 0.001,
            optimizer_type: OptimizerType::AdamW,
            weight_decay: 0.0,
            gradient_clip: None,
        }
    }
}

// ============================================================
// 训练状态
// ============================================================
#[derive(Debug, Clone, Default)]
pub struct TrainingState {
    pub epoch: u64,
    pub step: u64,
    pub loss: f32,
    pub learning_rate: f32,
}

// ============================================================
// 优化器状态 Trait
// ============================================================
pub trait OptimizerState: Send + Sync {
    fn update(
        &mut self,
        params: &mut [Tensor<f32>],
        grads: &[Tensor<f32>],
        config: &TrainerConfig,
    ) -> Result<(), String>;
    fn save(&self) -> Vec<u8>;
    fn load(&mut self, data: &[u8]) -> Result<(), String>;
}

// ============================================================
// SGD 优化器状态
// ============================================================
pub struct SGDOptimizerState {
    momentum: Vec<Tensor<f32>>,
    step: u64,
}

impl SGDOptimizerState {
    pub fn new(num_params: usize) -> Self {
        SGDOptimizerState {
            momentum: Vec::with_capacity(num_params),
            step: 0,
        }
    }
}

impl OptimizerState for SGDOptimizerState {
    fn update(
        &mut self,
        params: &mut [Tensor<f32>],
        grads: &[Tensor<f32>],
        config: &TrainerConfig,
    ) -> Result<(), String> {
        self.step += 1;
        let lr = config.learning_rate;
        let weight_decay = config.weight_decay;

        for i in 0..params.len() {
            let param_data = params[i].data_mut();
            let grad_data = grads[i].data();

            for j in 0..param_data.len() {
                let grad = grad_data[j];
                let decay = weight_decay * param_data[j];
                param_data[j] -= lr * (grad + decay);
            }
        }
        Ok(())
    }

    fn save(&self) -> Vec<u8> {
        Vec::new()
    }

    fn load(&mut self, _data: &[u8]) -> Result<(), String> {
        Ok(())
    }
}

// ============================================================
// AdamW 优化器状态 (简化版)
// ============================================================
pub struct AdamWOptimizerState {
    m: Vec<Tensor<f32>>,  // 一阶矩估计
    v: Vec<Tensor<f32>>,  // 二阶矩估计
    step: u64,
}

impl AdamWOptimizerState {
    pub fn new(params: &[Tensor<f32>]) -> Self {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());

        for p in params {
            m.push(Tensor::zeros(p.shape()));
            v.push(Tensor::zeros(p.shape()));
        }

        AdamWOptimizerState { m, v, step: 0 }
    }
}

impl OptimizerState for AdamWOptimizerState {
    fn update(
        &mut self,
        params: &mut [Tensor<f32>],
        grads: &[Tensor<f32>],
        config: &TrainerConfig,
    ) -> Result<(), String> {
        self.step += 1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let lr = config.learning_rate;
        let weight_decay = config.weight_decay;

        for i in 0..params.len() {
            let param_data = params[i].data_mut();
            let grad_data = grads[i].data();
            let m_data = self.m[i].data_mut();
            let v_data = self.v[i].data_mut();

            for j in 0..param_data.len() {
                let g = grad_data[j];

                // 权重衰减
                let decay = weight_decay * param_data[j];

                // 更新一阶矩
                m_data[j] = beta1 * m_data[j] + (1.0 - beta1) * (g + decay);
                // 更新二阶矩
                let g2 = (g + decay) * (g + decay);
                v_data[j] = beta2 * v_data[j] + (1.0 - beta2) * g2;

                // 偏差校正
                let m_hat = m_data[j] / (1.0 - beta1.powf(self.step as f32));
                let v_hat = v_data[j] / (1.0 - beta2.powf(self.step as f32));

                // 更新参数
                param_data[j] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
        Ok(())
    }

    fn save(&self) -> Vec<u8> {
        // TODO: 序列化优化器状态
        Vec::new()
    }

    fn load(&mut self, _data: &[u8]) -> Result<(), String> {
        Ok(())
    }
}

// ============================================================
// Trainer
// ============================================================
pub struct Trainer {
    graph: DagGraph,
    // 前向传播的值
    values: HashMap<u64, Tensor<f32>>,
    // 梯度
    grads: HashMap<u64, Tensor<f32>>,
    // 参数列表（所有需要训练的 Value ID）
    param_ids: Vec<u64>,
    // 优化器状态
    optimizer_state: Box<dyn OptimizerState>,
    // 训练状态
    training_state: TrainingState,
    // 配置
    config: TrainerConfig,
    // 内存池（复用内存）
    memory_pool: MemoryPool,
}

impl Trainer {
    pub fn new(
        graph: DagGraph,
        param_ids: Vec<u64>,
        config: TrainerConfig,
    ) -> Result<Self, String> {
        // 收集参数
        let mut params = Vec::new();
        for &id in &param_ids {
            if let Some(value) = graph.values.get(&id) {
                // 创建初始参数张量
                let shape: Vec<usize> = value.ty.shape.iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();
                let tensor = Tensor::zeros(&shape);
                params.push(tensor);
            } else {
                return Err(format!("Parameter {} not found", id));
            }
        }

        let optimizer_state: Box<dyn OptimizerState> = match config.optimizer_type {
            OptimizerType::SGD => {
                Box::new(SGDOptimizerState::new(param_ids.len()))
            }
            OptimizerType::AdamW => {
                Box::new(AdamWOptimizerState::new(&params))
            }
            OptimizerType::Adam => {
                // Adam 与 AdamW 类似，但权重衰减方式不同
                // 暂时用 AdamW
                Box::new(AdamWOptimizerState::new(&params))
            }
        };

        let memory_config = super::memory_reuse::MemoryConfig::training();

        Ok(Trainer {
            graph,
            values: HashMap::new(),
            grads: HashMap::new(),
            param_ids,
            optimizer_state,
            training_state: TrainingState::default(),
            config,
            memory_pool: MemoryPool::new(memory_config.block_size, memory_config.total_size),
        })
    }

    // ============================================================
    // 前向传播
    // ============================================================
    pub fn forward(&mut self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>, String> {
        self.values.clear();
        self.memory_pool.reset();

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

        // 加载常量（权重）
        for (&id, data) in &self.graph.constants {
            if let Some(value) = self.graph.values.get(&id) {
                let shape: Vec<usize> = value.ty.shape.iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();
                let tensor = Self::bytes_to_tensor(data, &shape)?;
                self.values.insert(id, tensor);
            }
        }

        let order = self.graph.topological_sort()?;
        for op_id in order {
            let op = self.graph.get_op(op_id)
                .ok_or_else(|| format!("Op {} not found", op_id))?
                .clone();
            self.execute_op(&op)?;
        }

        self.collect_outputs()
    }

    // ============================================================
    // 执行单个算子
    // ============================================================
    fn execute_op(&mut self, op: &crate::ir::dag::Op) -> Result<(), String> {
        let mut input_tensors = Vec::new();
        for &in_id in &op.inputs {
            if let Some(t) = self.values.get(&in_id) {
                input_tensors.push(t.clone());
            } else {
                return Err(format!("Input value {} not found for op {}", in_id, op.id));
            }
        }

        let outputs = super::executor::dispatch_op(&op.op_type, &input_tensors, &op.attrs)?;

        for (i, &out_id) in op.outputs.iter().enumerate() {
            if i < outputs.len() {
                let tensor = self.memory_pool.allocate_or_use(outputs[i].clone());
                self.values.insert(out_id, tensor);
            }
        }

        for &in_id in &op.inputs {
            let users = self.graph.get_users(in_id);
            if users.len() == 1 && users[0] == op.id {
                if let Some(tensor) = self.values.get(&in_id) {
                    self.memory_pool.mark_reusable(&tensor);
                }
            }
        }

        Ok(())
    }

    // ============================================================
    // 反向传播 (简化版)
    // ============================================================
    pub fn backward(&mut self, loss: &Tensor<f32>) -> Result<(), String> {
        // TODO: 实现完整的自动微分
        // 目前简化版：只记录 loss，梯度由用户手动计算
        self.training_state.loss = loss.data()[0];
        Ok(())
    }

    // ============================================================
    // 更新权重
    // ============================================================
    pub fn step(&mut self) -> Result<(), String> {
        if self.grads.is_empty() {
            return Err("No gradients available. Call backward() first.".to_string());
        }

        let mut params = Vec::new();
        let mut grads = Vec::new();

        for &param_id in &self.param_ids {
            if let Some(param) = self.values.get(&param_id) {
                params.push(param.clone());
            } else {
                return Err(format!("Parameter {} not found", param_id));
            }

            if let Some(grad) = self.grads.get(&param_id) {
                grads.push(grad.clone());
            } else {
                return Err(format!("Gradient for parameter {} not found", param_id));
            }
        }

        // 梯度裁剪
        if let Some(clip_val) = self.config.gradient_clip {
            let mut total_norm = 0.0;
            for g in &grads {
                for &v in g.data() {
                    total_norm += v * v;
                }
            }
            total_norm = total_norm.sqrt();
            if total_norm > clip_val {
                let scale = clip_val / total_norm;
                for g in &mut grads {
                    let data = g.data();
                    let mut new_data = data.to_vec();
                    for v in &mut new_data {
                        *v *= scale;
                    }
                    // 这里需要重新创建 Tensor
                }
            }
        }

        self.optimizer_state.update(&mut params, &grads, &self.config)?;

        // 更新 graph.constants（保存权重）
        for (i, &param_id) in self.param_ids.iter().enumerate() {
            if i < params.len() {
                let tensor = &params[i];
                let data = Self::tensor_to_bytes(tensor);
                self.graph.constants.insert(param_id, data);
            }
        }

        self.grads.clear();
        self.training_state.step += 1;

        Ok(())
    }

    // ============================================================
    // 设置梯度（外部注入）
    // ============================================================
    pub fn set_grad(&mut self, param_id: u64, grad: Tensor<f32>) {
        self.grads.insert(param_id, grad);
    }

    // ============================================================
    // 保存模型
    // ============================================================
    pub fn save(&self, path: &str, trainable: bool) -> Result<(), String> {
        let model_file = if trainable {
            crate::ir::serialize::ModelFile::new_trainable(
                &self.graph.name,
                "torch",
                self.graph.clone(),
                "adamw",
                self.config.learning_rate,
            )
        } else {
            crate::ir::serialize::ModelFile::new(&self.graph.name, "torch", self.graph.clone())
        };

        model_file.export(path)
    }

    // ============================================================
    // 工具函数
    // ============================================================
    fn bytes_to_tensor(data: &[u8], shape: &[usize]) -> Result<Tensor<f32>, String> {
        let float_data: Vec<f32> = data.chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(Tensor::new(float_data, shape))
    }

    fn tensor_to_bytes(tensor: &Tensor<f32>) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &v in tensor.data() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    // ============================================================
    // 获取训练状态
    // ============================================================
    pub fn get_training_state(&self) -> &TrainingState {
        &self.training_state
    }

    pub fn get_loss(&self) -> f32 {
        self.training_state.loss
    }

    pub fn set_loss(&mut self, loss: f32) {
        self.training_state.loss = loss;
    }

    // ============================================================
    // 收集输出
    // ============================================================
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