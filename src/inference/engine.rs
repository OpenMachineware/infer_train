// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Inference engine

use crate::backend::BackendRegistry;
use crate::memory::topology::MemoryTopology;
use crate::strategy::ExecutionStrategy;

// Inference engine
pub struct InferenceEngine {
    backends: BackendRegistry,
    strategy: ExecutionStrategy,
}

impl InferenceEngine {
    pub fn new() -> Self {
        let _topology = MemoryTopology::detect();

        Self {
            backends: BackendRegistry::new(),
            strategy: ExecutionStrategy::SingleDevice { device: 0 },
        }
    }

    pub fn register_backend(
        &mut self,
        backend: Box<dyn crate::backend::Backend>,
    ) {
        self.backends.register(backend);
    }

    pub fn set_strategy(&mut self, strategy: ExecutionStrategy) {
        self.strategy = strategy;
    }

    pub fn backends(&self) -> &BackendRegistry {
        &self.backends
    }

    pub fn strategy(&self) -> &ExecutionStrategy {
        &self.strategy
    }
}
