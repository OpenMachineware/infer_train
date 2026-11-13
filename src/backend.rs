// Backend trait 定义

pub mod arm;
pub mod cuda;
pub mod metal;

use crate::ops::{CoreOperator, OpError, QuantType};

/// Backend ID 类型
pub type BackendId = usize;

/// Backend trait
pub trait Backend: Send + Sync {
    /// 后端标识
    fn id(&self) -> usize;
    fn name(&self) -> &str;

    /// 内存信息
    fn available_memory(&self) -> usize;

    /// 能力查询
    fn supports_op(&self, op: &CoreOperator) -> bool;
    fn supports_quant(&self, quant: QuantType) -> bool;

    /// 执行算子
    fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> Result<(), OpError>;
}

/// Backend 注册表
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
