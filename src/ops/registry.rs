// src/ops/registry.rs

use crate::dtype::DType;
use crate::tensor::Tensor;
use std::collections::HashMap;

// ============================================================
// Attributes (operator parameters)
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
// Device types
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
// Operator trait (vendors only need to implement this)
// ============================================================

pub trait Operator<T: DType + Send + Sync>: Send + Sync {
    /// Operator name
    fn name(&self) -> &'static str;

    /// Forward pass
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T>;

    /// Backward pass (not supported by default)
    fn backward(
        &self,
        _grad_output: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }

    /// Whether quantized is supported
    fn supports_quantized(&self) -> bool {
        false
    }

    /// Device type
    fn device_type(&self) -> DeviceType {
        DeviceType::CPU
    }
}

// ============================================================
// Operator registry
// ============================================================

pub struct OperatorRegistry {
    ops: HashMap<String, Box<dyn Operator<f32>>>,
    // Can extend to other dtypes
}

impl OperatorRegistry {
    pub fn new() -> Self {
        OperatorRegistry { ops: HashMap::new() }
    }

    pub fn register<T: Operator<f32> + 'static>(&mut self, op: T) {
        self.ops.insert(op.name().to_string(), Box::new(op));
    }

    pub fn get(&self, name: &str) -> Option<&dyn Operator<f32>> {
        self.ops.get(name).map(|b| b.as_ref())
    }

    pub fn forward(
        &self,
        name: &str,
        inputs: &[&Tensor<f32>],
        attrs: &OpAttrs,
    ) -> Option<Tensor<f32>> {
        self.get(name).map(|op| op.forward(inputs, attrs))
    }
}
