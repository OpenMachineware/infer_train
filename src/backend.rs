// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Backend trait definition

pub mod arm;
pub mod cuda;
pub mod metal;

use crate::ops::{CoreOperator, OpError, QuantType};

// Backend ID type
pub type BackendId = usize;

// Backend trait
pub trait Backend: Send + Sync {
    /// Backend identifier
    fn id(&self) -> usize;
    fn name(&self) -> &str;

    /// Memory info
    fn available_memory(&self) -> usize;

    /// Capability query
    fn supports_op(&self, op: &CoreOperator) -> bool;
    fn supports_quant(&self, quant: QuantType) -> bool;

    /// Execute operator
    fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> Result<(), OpError>;
}

// Backend registry
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    pub fn find(&self, op: &CoreOperator) -> Option<&dyn Backend> {
        self.backends.iter().find(|b| b.supports_op(op)).map(|b| b.as_ref())
    }
}
