// 推理引擎

use crate::backend::BackendRegistry;
use crate::memory::topology::MemoryTopology;
use crate::strategy::ExecutionStrategy;

/// 推理引擎
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

    pub fn strategy(&self) -> &ExecutionStrategy {
        &self.strategy
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}
