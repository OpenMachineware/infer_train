pub mod ffi;
pub mod pytensor;

use pyo3::prelude::*;

#[pymodule]
fn infer_train_torch(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<pytensor::PyTensor>()?;
    Ok(())
}
