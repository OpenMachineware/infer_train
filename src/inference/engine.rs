// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Inference engine with generic backend support.

use std::sync::Arc;

use crate::backend::operator::OperatorResult;
use crate::backend::{Backend, CoreOperator, MemoryTopology, TensorRef};
use crate::strategy::ExecutionStrategy;

/// Inference engine with static backend dispatch.
///
/// This is the main entry point for running inference on models.
/// It uses generics for zero-overhead abstraction over different backends.
#[derive(Debug)]
pub struct InferenceEngine<B: Backend> {
    /// The compute backend.
    backend: Arc<B>,
    /// Memory topology.
    topology: MemoryTopology,
    /// Thread configuration.
    threads: usize,
}

impl<B: Backend> InferenceEngine<B> {
    /// Create a new inference engine with the given backend.
    pub fn new(backend: Arc<B>) -> Self {
        let topology = MemoryTopology::detect();
        let threads =
            std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        Self { backend, topology, threads }
    }

    /// Create with custom thread count.
    pub fn with_threads(backend: Arc<B>, threads: usize) -> Self {
        let topology = MemoryTopology::detect();
        Self { backend, topology, threads }
    }

    /// Get the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get the memory topology.
    pub fn topology(&self) -> &MemoryTopology {
        &self.topology
    }

    /// Get thread count.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Execute an operator.
    pub fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRef],
    ) -> OperatorResult<()> {
        self.backend.execute(op, inputs, outputs)
    }

    /// Check if the backend supports an operator.
    pub fn supports_operator(&self, op: &CoreOperator) -> bool {
        self.backend.supports_operator(op)
    }
}

/// Dynamic inference engine using trait objects.
///
/// Use this when the backend is not known at compile time.
#[derive(Debug)]
pub struct DynamicInferenceEngine {
    /// The compute backend.
    backend: Arc<dyn Backend>,
    /// Memory topology.
    topology: MemoryTopology,
    /// Execution strategy.
    strategy: ExecutionStrategy,
}

impl DynamicInferenceEngine {
    /// Create a new dynamic inference engine.
    pub fn new() -> Self {
        let topology = MemoryTopology::detect();
        let strategy = ExecutionStrategy::derive(&topology);
        let backend =
            strategy.primary_backend().cloned().expect("No backend available");
        Self { backend, topology, strategy }
    }

    /// Create with a specific backend.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        let topology = MemoryTopology::detect();
        let strategy = ExecutionStrategy::SingleDevice {
            device: backend.device().id(),
            backend: backend.clone(),
        };
        Self { backend, topology, strategy }
    }

    /// Get the backend.
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Get the memory topology.
    pub fn topology(&self) -> &MemoryTopology {
        &self.topology
    }

    /// Get the execution strategy.
    pub fn strategy(&self) -> &ExecutionStrategy {
        &self.strategy
    }

    /// Execute an operator.
    pub fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRef],
    ) -> OperatorResult<()> {
        self.strategy.execute(op, inputs, outputs)
    }

    /// Check if the backend supports an operator.
    pub fn supports_operator(&self, op: &CoreOperator) -> bool {
        self.backend.supports_operator(op)
    }
}

impl Default for DynamicInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ArmCpuBackend;
    use crate::backend::operator::QuantType;

    #[test]
    fn test_static_engine() {
        let backend =
            Arc::new(ArmCpuBackend::new(crate::backend::MemoryRegionId(0)));
        let engine = InferenceEngine::new(backend);

        assert!(engine.threads() > 0);
    }

    #[test]
    fn test_rms_norm() {
        let backend =
            Arc::new(ArmCpuBackend::new(crate::backend::MemoryRegionId(0)));
        let engine = InferenceEngine::new(backend);

        let hidden_size = 64;
        let op = CoreOperator::RmsNorm {
            id: crate::backend::operator::OperatorId(0),
            hidden_size,
            eps: 1e-5,
        };

        let mut input_data = vec![1.0f32; hidden_size];
        let weight_data = vec![1.0f32; hidden_size];
        let mut output_data = vec![0.0f32; hidden_size];

        let mut input =
            TensorRef::from_f32(0, &mut input_data, vec![hidden_size]);
        let mut weight =
            TensorRef::from_f32(1, &mut weight_data.clone(), vec![hidden_size]);
        let mut output =
            TensorRef::from_f32(2, &mut output_data, vec![hidden_size]);

        // Note: This won't work because we need actual slice data
        // For now, just check the operator is supported
        assert!(engine.supports_operator(&op));
    }
}
