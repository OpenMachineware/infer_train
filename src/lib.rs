// src/lib.rs

pub mod dtype;
pub mod tensor;
pub mod ops;
pub mod ir;
pub mod transform;
pub mod executor;
pub mod frontend;
pub mod autograd;
pub mod pytensor;
pub mod torch_bridge;

// 重新导出常用类型
pub use dtype::DType;
pub use tensor::Tensor;
pub use ops::math::add;

use pyo3::prelude::*;

// ============================================================
// PyTorch 插件
// ============================================================
#[cfg(feature = "torch")]
#[pymodule]
fn _infer_train_torch(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let rust_engine = PyModule::new(py, "rust_engine")?;

    // ---- 基础 class ----
    rust_engine.add_class::<pytensor::PyTensor>()?;
    rust_engine.add_class::<frontend::hook::HookTracer>()?;
    rust_engine.add_class::<ir::serialize::PyModelFile>()?;
    rust_engine.add_class::<ir::serialize::PyDagGraph>()?;
    rust_engine.add_class::<executor::executor::PyExecutor>()?;


    // ---- Frontend ----
    rust_engine.add_function(wrap_pyfunction!(frontend::gguf::import_gguf, &rust_engine)?)?;
    rust_engine.add_function(wrap_pyfunction!(frontend::gguf::export_gguf, &rust_engine)?)?;

    // ---- Torch Bridge ----
    rust_engine.add_function(wrap_pyfunction!(torch_bridge::trace_model, &rust_engine)?)?;
    rust_engine.add_function(wrap_pyfunction!(torch_bridge::trace_and_export, &rust_engine)?)?;
    rust_engine.add_function(wrap_pyfunction!(torch_bridge::trace_trainable, &rust_engine)?)?;
    rust_engine.add_function(wrap_pyfunction!(torch_bridge::load_and_infer, &rust_engine)?)?;

    // ---- IR ----
    rust_engine.add_class::<ir::serialize::PyModelFile>()?;
    rust_engine.add_class::<ir::serialize::PyDagGraph>()?;

    // ---- Executor ----
    rust_engine.add_class::<executor::PyExecutor>()?;

    m.add_submodule(&rust_engine)?;
    Ok(())
}
