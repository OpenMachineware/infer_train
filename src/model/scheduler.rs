// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Layer scheduling: CPU/GPU dispatch for inference

use crate::model::config::ModelConfig;
use crate::model::llama::{LlamaLayer, LlamaModel};
use crate::model::tensor::DataType;

/// Execution device
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

/// Layer execution plan
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub device: Device,
    pub quant_format: DataType,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        ExecutionPlan {
            device: Device::Cpu,
            quant_format: DataType::F32,
            batch_size: 1,
            seq_len: 1,
        }
    }
}

/// Scheduler: decides CPU vs GPU execution
pub struct Scheduler {
    config: ModelConfig,
    gpu_available: bool,
    prefer_gpu: bool,
}

impl Scheduler {
    pub fn new(config: ModelConfig) -> Self {
        let gpu_available =
            cfg!(target_arch = "aarch64") && cfg!(feature = "metal");

        Scheduler { config, gpu_available, prefer_gpu: true }
    }

    /// Decide execution device based on layer and context
    pub fn select_device(
        &self,
        layer: &LlamaLayer,
        plan: &ExecutionPlan,
    ) -> Device {
        if !self.gpu_available || !self.prefer_gpu {
            return Device::Cpu;
        }

        // Check quantization format
        let dtype = Self::layer_dtype(layer);
        match dtype {
            // GPU-native formats
            DataType::Q4_0
            | DataType::Q4_1
            | DataType::Q5_0
            | DataType::Q5_1
            | DataType::Q8_0
            | DataType::F16
            | DataType::BF16 => {
                // Large batch or long sequence: prefer GPU
                if plan.batch_size >= 1 || plan.seq_len >= 128 {
                    Device::Gpu
                } else {
                    Device::Cpu
                }
            }
            // K-quant: GPU optimized
            DataType::Q2_K
            | DataType::Q3_K
            | DataType::Q4_K
            | DataType::Q5_K
            | DataType::Q6_K => Device::Gpu,
            // Default: CPU
            _ => Device::Cpu,
        }
    }

    /// Create execution plan for model
    pub fn plan_execution(
        &self,
        model: &LlamaModel,
        batch_size: usize,
        seq_len: usize,
    ) -> Vec<(usize, Device)> {
        let mut plan = Vec::with_capacity(model.layers.len());

        let base_plan =
            ExecutionPlan { batch_size, seq_len, ..Default::default() };

        for (i, layer) in model.layers.iter().enumerate() {
            let device = self.select_device(layer, &base_plan);
            plan.push((i, device));
        }

        plan
    }

    /// Get layer's primary data type
    fn layer_dtype(layer: &LlamaLayer) -> DataType {
        // Check wq first (usually the largest weight)
        if let Some(ref t) = layer.wq {
            return t.dtype;
        }
        // Fallback to F32
        DataType::F32
    }

    /// Estimate memory requirement for layer
    pub fn estimate_memory(layer: &LlamaLayer, device: Device) -> usize {
        let tensor_bytes = layer.param_count();

        match device {
            Device::Gpu => {
                // GPU: quantized weights stay quantized, activations in F16
                tensor_bytes + (layer.param_count() / 2) // rough estimate
            }
            Device::Cpu => {
                // CPU: may need dequantized activations
                tensor_bytes * 4
            }
        }
    }

    /// Check if GPU has enough memory
    pub fn can_fit_gpu(&self, model: &LlamaModel) -> bool {
        if !self.gpu_available {
            return false;
        }

        // Rough estimate: model size + KV cache + activations
        let model_bytes: usize =
            model.layers.iter().map(|l| l.param_count()).sum::<usize>();
        let kv_cache_bytes = self.config.n_layers
            * self.config.n_kv_heads
            * self.config.max_position_embeddings
            * self.config.hidden_size
            / self.config.n_heads
            * 2; // K + V

        // Assume 8GB available (conservative for M1)
        let available_bytes = 8 * 1024 * 1024 * 1024;

        (model_bytes + kv_cache_bytes) < available_bytes
    }
}

/// Runtime: manages model execution
pub struct Runtime {
    scheduler: Scheduler,
    device: Device,
}

impl Runtime {
    pub fn new(config: ModelConfig) -> Self {
        let scheduler = Scheduler::new(config);
        let device =
            if scheduler.gpu_available { Device::Gpu } else { Device::Cpu };

        Runtime { scheduler, device }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}
