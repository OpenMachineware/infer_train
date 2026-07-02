use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::ffi::Tensor;

// ============================================================
// 辅助转换函数
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
// Python Tensor 类
// ============================================================
#[pyclass]
pub struct PyTensor {
    pub inner: Tensor,
}

#[pymethods]
impl PyTensor {
    // ============================================================
    // 构造函数
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
                inner: Tensor::new_f32(&data, &shape),
            }),
            "f64" => {
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let inner = Tensor::new_f64(&data_f64, &shape);
                Ok(PyTensor { inner })
            }
            "f16" => {
                let data_f16: Vec<u16> = data.iter().map(|&x| f32_to_f16(x)).collect();
                let inner = Tensor::new_f16(&data_f16, &shape);
                Ok(PyTensor { inner })
            }
            "bf16" => {
                let data_bf16: Vec<u16> = data.iter().map(|&x| f32_to_bf16(x)).collect();
                let inner = Tensor::new_bf16(&data_bf16, &shape);
                Ok(PyTensor { inner })
            }
            "i8" => {
                let scale = scale.ok_or_else(|| PyValueError::new_err("scale required for i8"))?;
                let zero_point = zero_point
                    .ok_or_else(|| PyValueError::new_err("zero_point required for i8"))?;
                let data_i8: Vec<i8> = data.iter().map(|&x| x as i8).collect();
                let inner = Tensor::new_quantized(&data_i8, &shape, scale, zero_point);
                Ok(PyTensor { inner })
            }
            _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}", dtype))),
        }
    }

    // ============================================================
    // 基础属性
    // ============================================================
    fn data(&self) -> Vec<f32> {
        match self.inner.dtype() {
            crate::ffi::it_dtype_t::IT_DTYPE_F32 => self.inner.data_as_f32().to_vec(),
            crate::ffi::it_dtype_t::IT_DTYPE_F64 => {
                self.inner.data_as_f64().iter().map(|&x| x as f32).collect()
            }
            crate::ffi::it_dtype_t::IT_DTYPE_F16 => {
                self.inner.data_as_f16().iter().map(|&x| f16_to_f32(x)).collect()
            }
            crate::ffi::it_dtype_t::IT_DTYPE_BF16 => {
                self.inner.data_as_bf16().iter().map(|&x| bf16_to_f32(x)).collect()
            }
            crate::ffi::it_dtype_t::IT_DTYPE_I8 => {
                self.inner.data_as_i8().iter().map(|&x| x as f32).collect()
            }
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    fn dtype(&self) -> String {
        match self.inner.dtype() {
            crate::ffi::it_dtype_t::IT_DTYPE_F32 => "f32".to_string(),
            crate::ffi::it_dtype_t::IT_DTYPE_F64 => "f64".to_string(),
            crate::ffi::it_dtype_t::IT_DTYPE_F16 => "f16".to_string(),
            crate::ffi::it_dtype_t::IT_DTYPE_BF16 => "bf16".to_string(),
            crate::ffi::it_dtype_t::IT_DTYPE_I8 => "i8".to_string(),
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

    fn __repr__(&self) -> String {
        format!(
            "PyTensor(shape={:?}, dtype={})",
            self.inner.shape(),
            self.dtype()
        )
    }

    // ============================================================
    // 已实现的算子
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

    #[pyo3(signature = (weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
    fn conv2d(
        &self,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> PyTensor {
        let bias_ref = bias.map(|b| &b.inner);
        PyTensor {
            inner: self.inner.conv2d(&weight.inner, bias_ref, stride, padding, dilation, groups),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn maxpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.maxpool2d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn avgpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.avgpool2d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (weight, bias, running_mean, running_var, eps=1e-5))]
    fn batchnorm2d(
        &self,
        weight: &PyTensor,
        bias: &PyTensor,
        running_mean: &PyTensor,
        running_var: &PyTensor,
        eps: f32,
    ) -> PyTensor {
        PyTensor {
            inner: self.inner.batchnorm2d(
                &weight.inner,
                &bias.inner,
                &running_mean.inner,
                &running_var.inner,
                eps,
            ),
        }
    }

    #[pyo3(signature = (weight, bias, eps=1e-5))]
    fn layernorm(&self, weight: &PyTensor, bias: &PyTensor, eps: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.layernorm(&weight.inner, &bias.inner, eps),
        }
    }

    #[pyo3(signature = (weight, eps=1e-6))]
    fn rmsnorm(&self, weight: &PyTensor, eps: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.rmsnorm(&weight.inner, eps),
        }
    }

    #[pyo3(signature = (weight, bias=None))]
    fn linear(&self, weight: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
        let bias_ref = bias.map(|b| &b.inner);
        PyTensor {
            inner: self.inner.linear(&weight.inner, bias_ref),
        }
    }

    fn transpose(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.transpose(),
        }
    }

    #[pyo3(signature = (dim, start, end, step=1))]
    fn slice(&self, dim: i32, start: i32, end: i32, step: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.slice(dim, start, end, step),
        }
    }

    #[pyo3(signature = (dims=Vec::new(), keepdim=false))]
    fn sum(&self, dims: Vec<i32>, keepdim: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.sum(&dims, keepdim),
        }
    }

    #[pyo3(signature = (dims=Vec::new(), keepdim=false))]
    fn mean(&self, dims: Vec<i32>, keepdim: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.mean(&dims, keepdim),
        }
    }

    fn max_all(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.max_all(),
        }
    }

    fn min_all(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.min_all(),
        }
    }

    fn prod_all(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.prod_all(),
        }
    }

    #[pyo3(signature = (unbiased=false))]
    fn var(&self, unbiased: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.var(unbiased),
        }
    }

    #[pyo3(signature = (unbiased=false))]
    fn std(&self, unbiased: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.std(unbiased),
        }
    }

    #[pyo3(signature = (dim=-1))]
    fn cumsum(&self, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.cumsum(dim),
        }
    }

    #[pyo3(signature = (dim=-1))]
    fn cumprod(&self, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.cumprod(dim),
        }
    }

    // ============================================================
    // 量化算子（已实现）
    // ============================================================
    fn quantized_add(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_add(&other.inner),
        }
    }

    fn quantized_matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_matmul(&other.inner),
        }
    }

    fn quantized_relu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_relu(),
        }
    }

    fn quantized_sigmoid(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_sigmoid(),
        }
    }
}
