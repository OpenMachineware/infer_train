// src/executor/trainer.rs

use std::collections::HashMap;

use crate::ir::dag::{DagGraph, Op};
use crate::tensor::Tensor;
use crate::ir::serialize::ModelFile;

use super::memory_reuse::{MemoryPool, MemoryConfig};
use super::amp::{AmpConfig, AmpGraphConverter};

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
// 训练器
// ============================================================

pub struct Trainer {
    graph: DagGraph,
    values: HashMap<u64, Tensor<f32>>,
    grads: HashMap<u64, Tensor<f32>>,
    param_ids: Vec<u64>,
    optimizer_state: Box<dyn OptimizerState>,
    training_state: TrainingState,
    config: TrainerConfig,
    memory_pool: MemoryPool,
    amp_config: AmpConfig,
    loss_scale: f32,
}

impl Trainer {
    pub fn new(
        graph: DagGraph,
        param_ids: Vec<u64>,
        config: TrainerConfig,
    ) -> Result<Self, String> {
        Self::with_amp(graph, param_ids, config, AmpConfig::inference())
    }

    pub fn with_amp(
        mut graph: DagGraph,
        param_ids: Vec<u64>,
        config: TrainerConfig,
        amp_config: AmpConfig,
    ) -> Result<Self, String> {
        let _amp_config_clone = amp_config.clone();
        let _loss_scale = amp_config.init_scale;

        // 转换 DAG 为混合精度
        if amp_config.enabled {
            AmpGraphConverter::convert(&mut graph, &amp_config, &param_ids)?;
        }

        // 创建优化器
        let optimizer_state: Box<dyn OptimizerState> = match config.optimizer_type {
            OptimizerType::SGD => {
                Box::new(SGDOptimizerState::new(param_ids.len()))
            }
            OptimizerType::AdamW => {
                let mut params = Vec::new();
                for &id in &param_ids {
                    if let Some(value) = graph.values.get(&id) {
                        let shape: Vec<usize> = value.ty.shape.iter()
                            .map(|&x| if x == -1 { 0 } else { x as usize })
                            .collect();
                        params.push(Tensor::zeros(&shape));
                    }
                }
                Box::new(AdamWOptimizerState::new(&params))
            }
            OptimizerType::Adam => {
                let mut params = Vec::new();
                for &id in &param_ids {
                    if let Some(value) = graph.values.get(&id) {
                        let shape: Vec<usize> = value.ty.shape.iter()
                            .map(|&x| if x == -1 { 0 } else { x as usize })
                            .collect();
                        params.push(Tensor::zeros(&shape));
                    }
                }
                Box::new(AdamWOptimizerState::new(&params))
            }
        };

        let memory_config = MemoryConfig::training();

        Ok(Trainer {
            graph,
            values: HashMap::new(),
            grads: HashMap::new(),
            param_ids,
            optimizer_state,
            training_state: TrainingState::default(),
            config,
            memory_pool: MemoryPool::new(memory_config.block_size, memory_config.total_size),
            amp_config,
            loss_scale: amp_config.init_scale,
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

        // 加载输入 (如果 AMP 启用，输入已经是 AMP dtype)
        for (i, &input_id) in self.graph.inputs.iter().enumerate() {
            self.values.insert(input_id, inputs[i].clone());
        }

        // 加载常量 (权重) - 保持在 FP32
        for (&id, data) in &self.graph.constants {
            if let Some(value) = self.graph.values.get(&id) {
                let shape: Vec<usize> = value.ty.shape.iter()
                    .map(|&x| if x == -1 { 0 } else { x as usize })
                    .collect();
                let tensor = Self::bytes_to_tensor(data, &shape)?;
                self.values.insert(id, tensor);
            }
        }

        // 执行
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

    fn execute_op(&mut self, op: &Op) -> Result<(), String> {
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
                let tensor = self.memory_pool.allocate_or_use(out_id, outputs[i].clone());
                self.values.insert(out_id, tensor);
            }
        }

        // 标记可复用
        for &in_id in &op.inputs {
            let users = self.graph.get_users(in_id);
            if users.len() == 1 && users[0] == op.id {
                if let Some(_tensor) = self.values.get(&in_id) {
                    self.memory_pool.mark_reusable(in_id);
                }
            }
        }

        Ok(())
    }

    // ============================================================
    // 反向传播
    // ============================================================

    pub fn backward(&mut self, loss: &Tensor<f32>) -> Result<(), String> {
        // 如果 AMP 启用，loss 已经是 AMP dtype
        // 反向传播使用 AMP dtype
        self.training_state.loss = loss.data()[0];

        // 如果 AMP 启用，缩放 loss
        let _loss_to_backward = if self.amp_config.enabled && self.amp_config.loss_scaling {
            // 缩放 loss
            let mut scaled = loss.clone();
            for v in scaled.data_mut() {
                *v *= self.loss_scale;
            }
            scaled
        } else {
            loss.clone()
        };

        // 从 loss 计算梯度
        // TODO: 实现完整的自动微分
        // 目前简化版

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
                // 如果 AMP 启用，梯度需要反缩放
                let mut grad_fp32 = grad.clone();
                if self.amp_config.enabled && self.amp_config.loss_scaling {
                    for v in grad_fp32.data_mut() {
                        *v /= self.loss_scale;
                    }
                }
                grads.push(grad_fp32);
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
                    for v in g.data_mut() {
                        *v *= scale;
                    }
                }
            }
        }

        self.optimizer_state.update(&mut params, &grads, &self.config)?;

        // 更新 graph.constants (保存权重)
        for (i, &param_id) in self.param_ids.iter().enumerate() {
            if i < params.len() {
                let tensor = &params[i];
                let data = Self::tensor_to_bytes(tensor);
                self.graph.constants.insert(param_id, data);
            }
        }

        self.grads.clear();
        self.training_state.step += 1;

        // 更新 loss_scale (动态调整)
        if self.amp_config.dynamic_scale {
            self.update_loss_scale();
        }

        Ok(())
    }

    // ============================================================
    // 损失缩放管理
    // ============================================================

    fn update_loss_scale(&mut self) {
        if !self.amp_config.enabled || !self.amp_config.dynamic_scale {
            return;
        }

        // 如果梯度有 inf/nan，减小 scale
        // 否则增大 scale
        let has_inf = self.grads.values().any(|g| {
            g.data().iter().any(|&v| v.is_infinite() || v.is_nan())
        });

        if has_inf {
            self.loss_scale *= 0.5;
            // 确保不低于最小值
            if self.loss_scale < 1.0 {
                self.loss_scale = 1.0;
            }
        } else {
            self.loss_scale *= 1.5;
            // 确保不超过最大值
            if self.loss_scale > 65536.0 {
                self.loss_scale = 65536.0;
            }
        }
    }

    // ============================================================
    // 保存模型
    // ============================================================

    pub fn save(&self, path: &str, trainable: bool) -> Result<(), String> {
        let model_file = if trainable {
            ModelFile::new_trainable(
                &self.graph.name,
                "torch",
                self.graph.clone(),
                "adamw",
                self.config.learning_rate,
            )
        } else {
            ModelFile::new(&self.graph.name, "torch", self.graph.clone())
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

    // ============================================================
    // Getters
    // ============================================================

    pub fn get_training_state(&self) -> &TrainingState {
        &self.training_state
    }

    pub fn get_loss(&self) -> f32 {
        self.training_state.loss
    }

    pub fn get_amp_config(&self) -> &AmpConfig {
        &self.amp_config
    }

    pub fn get_loss_scale(&self) -> f32 {
        self.loss_scale
    }

    pub fn set_grad(&mut self, param_id: u64, grad: Tensor<f32>) {
        self.grads.insert(param_id, grad);
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
// 优化器状态
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
// SGD 优化器
// ============================================================

pub struct SGDOptimizerState {
    #[allow(dead_code)]
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
// AdamW 优化器
// ============================================================

pub struct AdamWOptimizerState {
    m: Vec<Tensor<f32>>,
    v: Vec<Tensor<f32>>,
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
                let decay = weight_decay * param_data[j];

                m_data[j] = beta1 * m_data[j] + (1.0 - beta1) * (g + decay);
                let g2 = (g + decay) * (g + decay);
                v_data[j] = beta2 * v_data[j] + (1.0 - beta2) * g2;

                let m_hat = m_data[j] / (1.0 - beta1.powf(self.step as f32));
                let v_hat = v_data[j] / (1.0 - beta2.powf(self.step as f32));

                param_data[j] -= lr * m_hat / (v_hat.sqrt() + eps);
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
