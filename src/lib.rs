// src/lib.rs

pub mod dtype;
pub mod tensor;
pub mod ops;
pub mod ir;
pub mod transform;
pub mod executor;
pub mod frontend;

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
fn infer_train_torch(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // ---- Frontend ----
    m.add_function(wrap_pyfunction!(frontend::gguf::import_gguf, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::gguf::export_gguf, m)?)?;

    // ---- IR ----
    m.add_class::<ir::serialize::PyModelFile>()?;
    m.add_class::<ir::serialize::PyDagGraph>()?;

    // ---- Executor ----
    m.add_class::<executor::PyExecutor>()?;

    Ok(())
}
