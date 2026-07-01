// src/lib.rs
pub mod ffi;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;  // ← 添加这行

// ============================================================
// Python Tensor 包装
// ============================================================
#[pyclass]
pub struct PyTensor {
    inner: ffi::Tensor,
}

#[pymethods]
impl PyTensor {
    // ============================================================
    // FP32（默认）
    // ============================================================
    #[new]
    #[pyo3(signature = (data, shape, dtype="f32", scale=None, zero_point=None))]
    fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        dtype: &str,
        scale: Option<f32>,
        zero_point: Option<f32>,
    ) -> PyResult<Self> {
        match dtype {
            "f32" => Ok(PyTensor {
                inner: ffi::Tensor::new_f32(&data, &shape),
            }),
            "f64" => {
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let ptr = unsafe {
                    ffi::it_tensor_new(
                        data_f64.as_ptr() as *const std::os::raw::c_void,
                        shape.as_ptr(),
                        shape.len(),
                        ffi::it_dtype_t::IT_DTYPE_F64,
                    )
                };
                Ok(PyTensor {
                    inner: unsafe { ffi::Tensor::from_ptr(ptr) },
                })
            }
            "f16" => {
                let data_f16: Vec<u16> = data.iter().map(|&x| f32_to_f16(x)).collect();
                Ok(PyTensor {
                    inner: ffi::Tensor::new_f16(&data_f16, &shape),
                })
            }
            "bf16" => {
                let data_bf16: Vec<u16> = data.iter().map(|&x| f32_to_bf16(x)).collect();
                Ok(PyTensor {
                    inner: ffi::Tensor::new_bf16(&data_bf16, &shape),
                })
            }
            "i8" => {
                let scale = scale.ok_or_else(|| PyValueError::new_err("scale required for i8"))?;
                let zero_point = zero_point.ok_or_else(|| PyValueError::new_err("zero_point required for i8"))?;
                let data_i8: Vec<i8> = data.iter().map(|&x| x as i8).collect();
                Ok(PyTensor {
                    inner: ffi::Tensor::new_i8(&data_i8, &shape, scale, zero_point),
                })
            }
            _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}", dtype))),
        }
    }

    // ============================================================
    // 获取数据（自动转 f32）
    // ============================================================
    fn data(&self) -> Vec<f32> {
        match self.inner.dtype() {
            ffi::it_dtype_t::IT_DTYPE_F32 => {
                self.inner.data_as_f32().to_vec()
            }
            ffi::it_dtype_t::IT_DTYPE_F64 => {
                // 直接用 data_as_f64
                self.inner.data_as_f64().iter().map(|&x| x as f32).collect()
            }
            ffi::it_dtype_t::IT_DTYPE_F16 => {
                self.inner.data_as_f16().iter().map(|&x| f16_to_f32(x)).collect()
            }
            ffi::it_dtype_t::IT_DTYPE_BF16 => {
                self.inner.data_as_bf16().iter().map(|&x| bf16_to_f32(x)).collect()
            }
            ffi::it_dtype_t::IT_DTYPE_I8 => {
                self.inner.data_as_i8().iter().map(|&x| x as f32).collect()
            }
        }
    }

    // ============================================================
    // 其他方法
    // ============================================================
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    fn dtype(&self) -> String {
        match self.inner.dtype() {
            ffi::it_dtype_t::IT_DTYPE_F32 => "f32".to_string(),
            ffi::it_dtype_t::IT_DTYPE_F64 => "f64".to_string(),
            ffi::it_dtype_t::IT_DTYPE_F16 => "f16".to_string(),
            ffi::it_dtype_t::IT_DTYPE_BF16 => "bf16".to_string(),
            ffi::it_dtype_t::IT_DTYPE_I8 => "i8".to_string(),
        }
    }

    fn scale(&self) -> Option<f32> {
        if self.inner.is_quantized() {
            Some(self.inner.scale())
        } else {
            None
        }
    }

    fn zero_point(&self) -> Option<f32> {
        if self.inner.is_quantized() {
            Some(self.inner.zero_point())
        } else {
            None
        }
    }

    // ============================================================
    // 算子
    // ============================================================
    fn add(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.add(&other.inner),
        }
    }

    fn matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.matmul(&other.inner),
        }
    }

    fn relu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.relu(),
        }
    }

    fn softmax(&self, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.softmax(dim),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTensor(shape={:?}, dtype={}, data={:?})",
            self.inner.shape(),
            self.dtype(),
            self.data()
        )
    }
}

// ============================================================
// 辅助函数：f32 ↔ f16/bf16 转换
// ============================================================
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) - 127 + 15;
    let exp = if exp > 31 { 31 } else if exp < 0 { 0 } else { exp };
    let mant = (bits >> 13) & 0x3FF;
    (sign | (exp << 10) | mant) as u16
}

fn f16_to_f32(x: u16) -> f32 {
    let bits = (x as u32) << 16;
    f32::from_bits(bits)
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    (bits >> 16) as u16
}

fn bf16_to_f32(x: u16) -> f32 {
    let bits = (x as u32) << 16;
    f32::from_bits(bits)
}

// ============================================================
// Python 模块
// ============================================================
#[pymodule]
fn infer_train_torch(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
