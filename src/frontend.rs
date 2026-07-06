// src/frontend/mod.rs

pub mod hook;
pub mod jit_trace;

// 如果有其他前端
// #[cfg(feature = "tensorflow")]
// pub mod tf;
// 
// #[cfg(feature = "jax")]
// pub mod jax;

// 重新导出常用类型
pub use hook::HookTracer;
pub use jit_trace::{trace_from_torch, trace_and_save};
