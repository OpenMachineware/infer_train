pub mod ffi;
pub mod pytensor;

use pyo3::prelude::*;

#[cfg(feature = "torch")]
#[pymodule]
fn infer_train_torch(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    Ok(())
}

#[cfg(feature = "tensorflow")]
#[pymodule]
fn infer_train_tf(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    Ok(())
}

#[cfg(feature = "jax")]
#[pymodule]
fn infer_train_jax(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    Ok(())
}
