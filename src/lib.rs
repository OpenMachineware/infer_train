// src/lib.rs

pub mod autograd;
pub mod dtype;
pub mod executor;
pub mod frontend;
pub mod ir;
pub mod ops;
pub mod pytensor;
pub mod tensor;
pub mod torch_bridge;
pub mod transform;

// Re-export commonly used types
pub use dtype::DType;
pub use ops::math::add;
pub use tensor::Tensor;

use pyo3::prelude::*;

// ============================================================
// PyTorch Plugin
// ============================================================
#[cfg(feature = "torch")]
#[pymodule]
fn _infer_train_torch(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let rust_engine = PyModule::new(py, "rust_engine")?;

    // ---- Basic classes ----
    rust_engine.add_class::<pytensor::PyTensor>()?;
    rust_engine.add_class::<frontend::hook::HookTracer>()?;
    rust_engine.add_class::<ir::serialize::PyModelFile>()?;
    rust_engine.add_class::<ir::serialize::PyDagGraph>()?;
    rust_engine.add_class::<executor::executor::PyExecutor>()?;

    // ---- Frontend ----
    rust_engine.add_function(wrap_pyfunction!(
        frontend::gguf::import_gguf,
        &rust_engine
    )?)?;
    rust_engine.add_function(wrap_pyfunction!(
        frontend::gguf::export_gguf,
        &rust_engine
    )?)?;

    // ---- Torch Bridge ----
    rust_engine.add_function(wrap_pyfunction!(
        torch_bridge::trace_model,
        &rust_engine
    )?)?;
    rust_engine.add_function(wrap_pyfunction!(
        torch_bridge::trace_and_export,
        &rust_engine
    )?)?;
    rust_engine.add_function(wrap_pyfunction!(
        torch_bridge::trace_trainable,
        &rust_engine
    )?)?;
    rust_engine.add_function(wrap_pyfunction!(
        torch_bridge::load_and_infer,
        &rust_engine
    )?)?;

    // ---- IR ----
    rust_engine.add_class::<ir::serialize::PyModelFile>()?;
    rust_engine.add_class::<ir::serialize::PyDagGraph>()?;

    // ---- Executor ----
    rust_engine.add_class::<executor::PyExecutor>()?;

    m.add_submodule(&rust_engine)?;
    Ok(())
}
