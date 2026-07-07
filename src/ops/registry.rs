// src/ops/registry.rs

use crate::dtype::DType;
use crate::tensor::Tensor;
use std::collections::HashMap;

// ============================================================
// 属性（算子的参数）
// ============================================================

#[derive(Debug, Clone, Default)]
pub struct OpAttrs {
    pub ints: HashMap<String, i64>,
    pub floats: HashMap<String, f32>,
    pub bools: HashMap<String, bool>,
    pub strings: HashMap<String, String>,
    pub int_lists: HashMap<String, Vec<i64>>,
    pub float_lists: HashMap<String, Vec<f32>>,
}

impl OpAttrs {
    pub fn new() -> Self {
        OpAttrs::default()
    }

    pub fn with_int(mut self, key: &str, val: i64) -> Self {
        self.ints.insert(key.to_string(), val);
        self
    }

    pub fn with_float(mut self, key: &str, val: f32) -> Self {
        self.floats.insert(key.to_string(), val);
        self
    }

    pub fn with_bool(mut self, key: &str, val: bool) -> Self {
        self.bools.insert(key.to_string(), val);
        self
    }

    pub fn with_int_list(mut self, key: &str, val: Vec<i64>) -> Self {
        self.int_lists.insert(key.to_string(), val);
        self
    }

    pub fn with_float_list(mut self, key: &str, val: Vec<f32>) -> Self {
        self.float_lists.insert(key.to_string(), val);
        self
    }

    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.ints.get(key).copied()
    }

    pub fn get_float(&self, key: &str) -> Option<f32> {
        self.floats.get(key).copied()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.bools.get(key).copied()
    }

    pub fn get_int_list(&self, key: &str) -> Option<&Vec<i64>> {
        self.int_lists.get(key)
    }

    pub fn get_float_list(&self, key: &str) -> Option<&Vec<f32>> {
        self.float_lists.get(key)
    }

    pub fn get_string(&self, key: &str) -> Option<String> {
        self.strings.get(key).cloned()
    }
}

// ============================================================
// 设备类型
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    CPU,
    NPU,
    GPU,
    TPU,
    Custom(&'static str),
}

// ============================================================
// 算子 Trait（厂商只需实现这个）
// ============================================================

pub trait Operator<T: DType + Send + Sync>: Send + Sync {
    /// 算子名称
    fn name(&self) -> &'static str;

    /// 前向传播
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T>;

    /// 反向传播（默认不支持）
    fn backward(
        &self,
        _grad_output: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }

    /// 是否支持量化
    fn supports_quantized(&self) -> bool {
        false
    }

    /// 设备类型
    fn device_type(&self) -> DeviceType {
        DeviceType::CPU
    }
}

// ============================================================
// 算子注册表
// ============================================================

pub struct OperatorRegistry {
    ops: HashMap<String, Box<dyn Operator<f32>>>,
    // 可以扩展其他 dtype
}

impl OperatorRegistry {
    pub fn new() -> Self {
        OperatorRegistry {
            ops: HashMap::new(),
        }
    }

    pub fn register<T: Operator<f32> + 'static>(&mut self, op: T) {
        self.ops.insert(op.name().to_string(), Box::new(op));
    }

    pub fn get(&self, name: &str) -> Option<&dyn Operator<f32>> {
        self.ops.get(name).map(|b| b.as_ref())
    }

    pub fn forward(&self, name: &str, inputs: &[&Tensor<f32>], attrs: &OpAttrs) -> Option<Tensor<f32>> {
        self.get(name).map(|op| op.forward(inputs, attrs))
    }
}
