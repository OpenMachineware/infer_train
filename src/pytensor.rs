use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::Py;
use pyo3::pyclass;
use pyo3::types::PyList;
use pyo3::PyResult;
use crate::ffi::Tensor;
use crate::ffi::it_tensor;
use crate::ffi::it_cat;
use crate::ffi::it_where;

// ============================================================
// 辅助转换函数
// ============================================================
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) - 127 + 15;
    let exp = exp.clamp(0, 31);
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
#[derive(Clone)]
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
        if self.inner.as_ptr().is_null() {
            return Vec::new();
        }
        let result = match self.inner.dtype() {
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
        };
        result
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
            let s = self.inner.scale();
            Some(s)
        } else {
            None
        }
    }

    fn zero_point(&self) -> Option<f32> {
        if self.inner.is_quantized() {
            let z = self.inner.zero_point();
            Some(z)
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
    // 数学算子
    // ============================================================
    fn add(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.add(&other.inner),
        }
    }

    fn sub(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.sub(&other.inner),
        }
    }

    fn mul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.mul(&other.inner),
        }
    }

    fn div(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.div(&other.inner),
        }
    }

    fn pow(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.pow(&other.inner),
        }
    }

    fn exp(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.exp(),
        }
    }

    fn sqrt(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.sqrt(),
        }
    }

    fn log(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.log(),
        }
    }

    fn log2(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.log2(),
        }
    }

    fn log10(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.log10(),
        }
    }

    fn abs(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.abs(),
        }
    }

    fn neg(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.neg(),
        }
    }

    #[pyo3(signature = (min_val, max_val))]
    fn clamp(&self, min_val: f32, max_val: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.clamp(min_val, max_val),
        }
    }

    fn floor(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.floor(),
        }
    }

    fn ceil(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.ceil(),
        }
    }

    fn round(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.round(),
        }
    }

    fn eq(&self, other: &PyTensor) -> Vec<u8> {
        self.inner.eq(&other.inner)
    }

    fn ne(&self, other: &PyTensor) -> Vec<u8> {
        self.inner.ne(&other.inner)
    }

    fn gt(&self, other: &PyTensor) -> Vec<u8> {
        self.inner.gt(&other.inner)
    }

    fn lt(&self, other: &PyTensor) -> Vec<u8> {
        self.inner.lt(&other.inner)
    }

    fn ge(&self, other: &PyTensor) -> Vec<u8> {
        self.inner.ge(&other.inner)
    }

    fn le(&self, other: &PyTensor) -> Vec<u8> {
        self.inner.le(&other.inner)
    }

    fn reshape(&self, new_shape: Vec<usize>) -> PyTensor {
        PyTensor {
            inner: self.inner.reshape(&new_shape),
        }
    }

    #[pyo3(signature = (dims=Vec::new(), keepdim=false))]
    fn quantized_sum(&self, dims: Vec<i32>, keepdim: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_sum(&dims, keepdim),
        }
    }

    #[pyo3(signature = (dims=Vec::new(), keepdim=false))]
    fn quantized_mean(&self, dims: Vec<i32>, keepdim: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_mean(&dims, keepdim),
        }
    }

    fn quantized_max_all(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_max_all(),
        }
    }

    fn quantized_min_all(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_min_all(),
        }
    }

    fn quantized_matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_matmul(&other.inner),
        }
    }

    // ============================================================
    // 规约算子
    // ============================================================
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

    fn argmax(&self) -> Vec<i64> {
        self.inner.argmax()
    }

    fn argmin(&self) -> Vec<i64> {
        self.inner.argmin()
    }

    #[pyo3(signature = (k, dim=-1, largest=true))]
    fn topk(&self, k: usize, dim: i32, largest: bool) -> (PyTensor, PyTensor) {
        let (values, indices) = self.inner.topk(k, dim, largest);
        (PyTensor { inner: values }, PyTensor { inner: indices })
    }

    // ============================================================
    // 矩阵算子
    // ============================================================
    fn matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.matmul(&other.inner),
        }
    }

    fn batch_matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.batch_matmul(&other.inner),
        }
    }

    fn vec_matmul(&self, mat: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.vec_matmul(&mat.inner),
        }
    }

    fn transpose(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.transpose(),
        }
    }

    fn quantized_vec_matmul(&self, mat: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_vec_matmul(&mat.inner),
        }
    }

    fn quantized_transpose(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_transpose(),
        }
    }

    // ============================================================
    // NN 算子
    // ============================================================
    #[pyo3(signature = (weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
    fn conv1d(
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
            inner: self.inner.conv1d(&weight.inner, bias_ref, stride, padding, dilation, groups),
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

    #[pyo3(signature = (weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
    fn conv3d(
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
            inner: self.inner.conv3d(&weight.inner, bias_ref, stride, padding, dilation, groups),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn maxpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.maxpool1d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn maxpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.maxpool2d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn maxpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.maxpool3d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn avgpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.avgpool1d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn avgpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.avgpool2d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn avgpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.avgpool3d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (weight, bias, running_mean, running_var, eps=1e-5))]
    fn batchnorm1d(
        &self,
        weight: &PyTensor,
        bias: &PyTensor,
        running_mean: &PyTensor,
        running_var: &PyTensor,
        eps: f32,
    ) -> PyTensor {
        PyTensor {
            inner: self.inner.batchnorm1d(
                &weight.inner,
                &bias.inner,
                &running_mean.inner,
                &running_var.inner,
                eps,
            ),
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

    #[pyo3(signature = (weight, bias, eps=1e-5))]
    fn instancenorm2d(&self, weight: &PyTensor, bias: &PyTensor, eps: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.instancenorm2d(&weight.inner, &bias.inner, eps),
        }
    }

    #[pyo3(signature = (weight, bias, num_groups, eps=1e-5))]
    fn groupnorm(&self, weight: &PyTensor, bias: &PyTensor, num_groups: i32, eps: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.groupnorm(&weight.inner, &bias.inner, num_groups, eps),
        }
    }

    #[pyo3(signature = (weight, bias=None))]
    fn linear(&self, weight: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
        let bias_ref = bias.map(|b| &b.inner);
        PyTensor {
            inner: self.inner.linear(&weight.inner, bias_ref),
        }
    }

    #[pyo3(signature = (indices, padding_idx=-1))]
    fn embedding(&self, indices: Vec<i64>, padding_idx: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.embedding(&indices, padding_idx),
        }
    }

    #[pyo3(signature = (p=0.5, training=false, seed=0))]
    fn dropout(&self, p: f32, training: bool, seed: u32) -> PyTensor {
        PyTensor {
            inner: self.inner.dropout(p, training, seed),
        }
    }

    // ============================================================
    // 激活函数
    // ============================================================
    fn relu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.relu(),
        }
    }

    #[pyo3(signature = (alpha=0.01))]
    fn leaky_relu(&self, alpha: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.leaky_relu(alpha),
        }
    }

    #[pyo3(signature = (alpha=1.0))]
    fn elu(&self, alpha: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.elu(alpha),
        }
    }

    fn gelu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.gelu(),
        }
    }

    fn relu6(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.relu6(),
        }
    }

    fn sigmoid(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.sigmoid(),
        }
    }

    fn tanh(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.tanh(),
        }
    }

    fn silu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.silu(),
        }
    }

    fn hard_swish(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.hard_swish(),
        }
    }

    fn hard_sigmoid(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.hard_sigmoid(),
        }
    }

    #[pyo3(signature = (beta=1.0, threshold=20.0))]
    fn softplus(&self, beta: f32, threshold: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.softplus(beta, threshold),
        }
    }

    #[pyo3(signature = (lambda=0.5))]
    fn softshrink(&self, lambda: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.softshrink(lambda),
        }
    }

    #[pyo3(signature = (alpha=1.0))]
    fn celu(&self, alpha: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.celu(alpha),
        }
    }

    #[pyo3(signature = (dim=-1))]
    fn softmax(&self, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.softmax(dim),
        }
    }

    #[pyo3(signature = (dim=-1))]
    fn log_softmax(&self, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.log_softmax(dim),
        }
    }

    // ============================================================
    // 张量操作
    // ============================================================
    #[pyo3(signature = (dim, start, end, step=1))]
    fn slice(&self, dim: i32, start: i32, end: i32, step: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.slice(dim, start, end, step),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (condition, condition_shape, true_val, false_val))]
    fn where_op(
        condition: Vec<u8>,
        condition_shape: Vec<usize>,
        true_val: &PyTensor,
        false_val: &PyTensor,
    ) -> PyResult<PyTensor> {
        let ptr = unsafe {
            it_where(
                condition.as_ptr(),
                condition_shape.as_ptr(),
                condition_shape.len(),
                true_val.inner.as_ptr(),
                false_val.inner.as_ptr(),
            )
        };
        if ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("where failed"));
        }
        Ok(PyTensor {
            inner: unsafe { Tensor::from_ptr(ptr) },
        })
    }

    #[pyo3(signature = (indices, indices_shape, dim=0))]
    fn gather(&self, indices: Vec<i64>, indices_shape: Vec<usize>, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.gather(&indices, &indices_shape, dim),
        }
    }

    #[pyo3(signature = (indices, indices_shape, src, dim=0))]
    fn scatter(&self, indices: Vec<i64>, indices_shape: Vec<usize>, src: &PyTensor, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.scatter(&indices, &indices_shape, &src.inner, dim),
        }
    }

    #[pyo3(signature = (dim=-1, ascending=true))]
    fn sort(&self, dim: i32, ascending: bool) -> (PyTensor, PyTensor) {
        let (values, indices) = self.inner.sort(dim, ascending);
        (PyTensor { inner: values }, PyTensor { inner: indices })
    }

    // ============================================================
    // cat（静态方法）
    // ============================================================
    #[staticmethod]
    fn cat(tensors: &Bound<PyList>, dim: i32, py: Python) -> PyResult<PyTensor> {
        let mut ptrs: Vec<*const it_tensor> = Vec::new();
        for item in tensors.iter() {
            let pytensor: Py<PyTensor> = item.extract()?;
            let borrowed = pytensor.borrow(py);
            ptrs.push(borrowed.inner.as_ptr());
        }
        let ptr = unsafe {
            it_cat(
                ptrs.as_ptr(),
                ptrs.len(),
                dim,
            )
        };
        if ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("cat failed"));
        }
        Ok(PyTensor {
            inner: unsafe { Tensor::from_ptr(ptr) },
        })
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
    // 量化算子
    // ============================================================

    fn quantized_add(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_add(&other.inner),
        }
    }

    fn quantized_sub(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_sub(&other.inner),
        }
    }

    fn quantized_mul(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_mul(&other.inner),
        }
    }

    fn quantized_div(&self, other: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_div(&other.inner),
        }
    }

    fn quantized_exp(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_exp(),
        }
    }

    fn quantized_sqrt(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_sqrt(),
        }
    }

    fn quantized_abs(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_abs(),
        }
    }

    fn quantized_neg(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_neg(),
        }
    }

    #[pyo3(signature = (min_val, max_val))]
    fn quantized_clamp(&self, min_val: f32, max_val: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_clamp(min_val, max_val),
        }
    }

    #[pyo3(signature = (weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
    fn quantized_conv1d(
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
            inner: self.inner.quantized_conv1d(&weight.inner, bias_ref, stride, padding, dilation, groups),
        }
    }

    #[pyo3(signature = (weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
    fn quantized_conv2d(
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
            inner: self.inner.quantized_conv2d(&weight.inner, bias_ref, stride, padding, dilation, groups),
        }
    }

    #[pyo3(signature = (weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
    fn quantized_conv3d(
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
            inner: self.inner.quantized_conv3d(&weight.inner, bias_ref, stride, padding, dilation, groups),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn quantized_maxpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_maxpool1d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn quantized_maxpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_maxpool2d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn quantized_maxpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_maxpool3d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn quantized_avgpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_avgpool1d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn quantized_avgpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_avgpool2d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (kernel_size, stride=-1, padding=0))]
    fn quantized_avgpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_avgpool3d(kernel_size, stride, padding),
        }
    }

    #[pyo3(signature = (weight, bias, running_mean, running_var, eps=1e-5))]
    fn quantized_batchnorm1d(
        &self,
        weight: &PyTensor,
        bias: &PyTensor,
        running_mean: &PyTensor,
        running_var: &PyTensor,
        eps: f32,
    ) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_batchnorm1d(
                &weight.inner,
                &bias.inner,
                &running_mean.inner,
                &running_var.inner,
                eps,
            ),
        }
    }

    #[pyo3(signature = (weight, bias, running_mean, running_var, eps=1e-5))]
    fn quantized_batchnorm2d(
        &self,
        weight: &PyTensor,
        bias: &PyTensor,
        running_mean: &PyTensor,
        running_var: &PyTensor,
        eps: f32,
    ) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_batchnorm2d(
                &weight.inner,
                &bias.inner,
                &running_mean.inner,
                &running_var.inner,
                eps,
            ),
        }
    }

    #[pyo3(signature = (weight, bias, eps=1e-5))]
    fn quantized_layernorm(&self, weight: &PyTensor, bias: &PyTensor, eps: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_layernorm(&weight.inner, &bias.inner, eps),
        }
    }

    #[pyo3(signature = (weight, eps=1e-6))]
    fn quantized_rmsnorm(&self, weight: &PyTensor, eps: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_rmsnorm(&weight.inner, eps),
        }
    }

    #[pyo3(signature = (weight, bias=None))]
    fn quantized_linear(&self, weight: &PyTensor, bias: Option<&PyTensor>) -> PyTensor {
        let bias_ref = bias.map(|b| &b.inner);
        PyTensor {
            inner: self.inner.quantized_linear(&weight.inner, bias_ref),
        }
    }

    #[pyo3(signature = (indices, padding_idx=-1))]
    fn quantized_embedding(&self, indices: Vec<i64>, padding_idx: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_embedding(&indices, padding_idx),
        }
    }

    fn quantized_relu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_relu(),
        }
    }

    #[pyo3(signature = (alpha=0.01))]
    fn quantized_leaky_relu(&self, alpha: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_leaky_relu(alpha),
        }
    }

    #[pyo3(signature = (alpha=1.0))]
    fn quantized_elu(&self, alpha: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_elu(alpha),
        }
    }

    fn quantized_gelu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_gelu(),
        }
    }

    fn quantized_relu6(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_relu6(),
        }
    }

    fn quantized_sigmoid(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_sigmoid(),
        }
    }

    fn quantized_tanh(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_tanh(),
        }
    }

    fn quantized_silu(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_silu(),
        }
    }

    fn quantized_hard_swish(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_hard_swish(),
        }
    }

    fn quantized_hard_sigmoid(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_hard_sigmoid(),
        }
    }

    #[pyo3(signature = (dim=-1))]
    fn quantized_softmax(&self, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_softmax(dim),
        }
    }

    #[pyo3(signature = (dim=-1))]
    fn quantized_log_softmax(&self, dim: i32) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_log_softmax(dim),
        }
    }

    // ============================================================
    // 注意力算子
    // ============================================================
    #[pyo3(signature = (key, value, mask=None, scale=-1.0, is_causal=false, dropout_p=0.0))]
    fn scaled_dot_product_attention(
        &self,
        key: &PyTensor,
        value: &PyTensor,
        mask: Option<&PyTensor>,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> PyTensor {
        let mask_ref = mask.map(|m| &m.inner);
        PyTensor {
            inner: self.inner.scaled_dot_product_attention(
                &key.inner,
                &value.inner,
                mask_ref,
                scale,
                is_causal,
                dropout_p,
            ),
        }
    }

    #[pyo3(signature = (key, value, mask=None, num_heads=8, scale=-1.0, is_causal=false, dropout_p=0.0))]
    fn multi_head_attention(
        &self,
        key: &PyTensor,
        value: &PyTensor,
        mask: Option<&PyTensor>,
        num_heads: i32,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> PyTensor {
        let mask_ref = mask.map(|m| &m.inner);
        PyTensor {
            inner: self.inner.multi_head_attention(
                &key.inner,
                &value.inner,
                mask_ref,
                num_heads,
                scale,
                is_causal,
                dropout_p,
            ),
        }
    }

    #[pyo3(signature = (cos, sin))]
    fn rotary_embedding(&self, cos: &PyTensor, sin: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.rotary_embedding(&cos.inner, &sin.inner),
        }
    }

    #[pyo3(signature = (key, value, mask=None, scale=-1.0, is_causal=false, dropout_p=0.0))]
    fn quantized_scaled_dot_product_attention(
        &self,
        key: &PyTensor,
        value: &PyTensor,
        mask: Option<&PyTensor>,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> PyTensor {
        let mask_ref = mask.map(|m| &m.inner);
        PyTensor {
            inner: self.inner.quantized_scaled_dot_product_attention(
                &key.inner,
                &value.inner,
                mask_ref,
                scale,
                is_causal,
                dropout_p,
            ),
        }
    }

    #[pyo3(signature = (key, value, mask=None, num_heads=8, scale=-1.0, is_causal=false, dropout_p=0.0))]
    fn quantized_multi_head_attention(
        &self,
        key: &PyTensor,
        value: &PyTensor,
        mask: Option<&PyTensor>,
        num_heads: i32,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> PyTensor {
        let mask_ref = mask.map(|m| &m.inner);
        PyTensor {
            inner: self.inner.quantized_multi_head_attention(
                &key.inner,
                &value.inner,
                mask_ref,
                num_heads,
                scale,
                is_causal,
                dropout_p,
            ),
        }
    }

    #[pyo3(signature = (cos, sin))]
    fn quantized_rotary_embedding(&self, cos: &PyTensor, sin: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.quantized_rotary_embedding(&cos.inner, &sin.inner),
        }
    }

    // ============================================================
    // 损失函数
    // ============================================================
    #[pyo3(signature = (target, reduction=true))]
    fn cross_entropy_loss(&self, target: Vec<i64>, reduction: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.cross_entropy_loss(&target, reduction),
        }
    }

    #[pyo3(signature = (target, reduction=true))]
    fn mse_loss(&self, target: &PyTensor, reduction: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.mse_loss(&target.inner, reduction),
        }
    }

    #[pyo3(signature = (target, reduction=true))]
    fn l1_loss(&self, target: &PyTensor, reduction: bool) -> PyTensor {
        PyTensor {
            inner: self.inner.l1_loss(&target.inner, reduction),
        }
    }

    #[pyo3(signature = (target, reduction=true, eps=1e-7))]
    fn bce_loss(&self, target: &PyTensor, reduction: bool, eps: f32) -> PyTensor {
        PyTensor {
            inner: self.inner.bce_loss(&target.inner, reduction, eps),
        }
    }
}

// ============================================================
// 优化器 Python 绑定
// ============================================================
// ---------- AdamState ----------
unsafe impl Send for AdamState {}
unsafe impl Sync for AdamState {}

#[pyclass]
pub struct AdamState {
    inner: crate::ffi::AdamState,
}

#[pymethods]
impl AdamState {
    #[new]
    fn new(
        py: Python,
        params: Vec<Py<PyTensor>>,
        param_shapes: Vec<usize>,
        param_ndims: Vec<usize>,
    ) -> Self {
        let mut tensors: Vec<Tensor> = Vec::new();
        for p in params.iter() {
            let pytensor = p.borrow(py);
            tensors.push(pytensor.inner.clone());
        }
        AdamState {
            inner: crate::ffi::AdamState::new(&tensors, &param_shapes, &param_ndims),
        }
    }

    fn update(
        &mut self,
        py: Python,
        params: Vec<Py<PyTensor>>,
        grads: Vec<Py<PyTensor>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        let mut param_tensors: Vec<Tensor> = Vec::new();
        let mut grad_tensors: Vec<Tensor> = Vec::new();

        for p in params.iter() {
            let pytensor = p.borrow(py);
            param_tensors.push(pytensor.inner.clone());
        }
        for g in grads.iter() {
            let grad_tensor = g.borrow(py);
            grad_tensors.push(grad_tensor.inner.clone());
        }

        self.inner.update(
            &mut param_tensors,
            &grad_tensors,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        );

        // 写回 params
        for (i, p) in params.iter().enumerate() {
            let mut pytensor = p.borrow_mut(py);
            pytensor.inner = param_tensors[i].clone();
        }
    }
}

// ---------- AdamWState ----------
unsafe impl Send for AdamWState {}
unsafe impl Sync for AdamWState {}

#[pyclass]
pub struct AdamWState {
    inner: crate::ffi::AdamWState,
}

#[pymethods]
impl AdamWState {
    #[new]
    fn new(
        py: Python,
        params: Vec<Py<PyTensor>>,
        param_shapes: Vec<usize>,
        param_ndims: Vec<usize>,
    ) -> Self {
        let mut tensors: Vec<Tensor> = Vec::new();
        for p in params.iter() {
            let pytensor = p.borrow(py);
            tensors.push(pytensor.inner.clone());
        }
        AdamWState {
            inner: crate::ffi::AdamWState::new(&tensors, &param_shapes, &param_ndims),
        }
    }

    fn update(
        &mut self,
        py: Python,
        params: Vec<Py<PyTensor>>,
        grads: Vec<Py<PyTensor>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        let mut param_tensors: Vec<Tensor> = Vec::new();
        let mut grad_tensors: Vec<Tensor> = Vec::new();

        for p in params.iter() {
            let pytensor = p.borrow(py);
            param_tensors.push(pytensor.inner.clone());
        }
        for g in grads.iter() {
            let grad_tensor = g.borrow(py);
            grad_tensors.push(grad_tensor.inner.clone());
        }

        self.inner.update(
            &mut param_tensors,
            &grad_tensors,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        );

        for (i, p) in params.iter().enumerate() {
            let mut pytensor = p.borrow_mut(py);
            pytensor.inner = param_tensors[i].clone();
        }
    }
}

// ============================================================
// 全局函数：SGD 更新
// ============================================================
#[pyfunction]
pub fn sgd_update(
    py: Python,
    params: Vec<Py<PyTensor>>,
    grads: Vec<Py<PyTensor>>,
    _lr: f32,
    _momentum: f32,
    _weight_decay: f32,
    _esterov: bool,
) -> PyResult<()> {
    let mut param_tensors: Vec<Tensor> = Vec::new();
    let mut grad_tensors: Vec<Tensor> = Vec::new();

    for p in params.iter() {
        let pytensor = p.borrow(py);
        param_tensors.push(pytensor.inner.clone());
    }
    for g in grads.iter() {
        let grad_tensor = g.borrow(py);
        grad_tensors.push(grad_tensor.inner.clone());
    }

    // 调用 ff::sgd_update
    // 需要在 impl Tensor 中加 sgd_update 方法或直接 C API

    // 写回参数
    for (i, p) in params.iter().enumerate() {
        let mut pytensor = p.borrow_mut(py);
        pytensor.inner = param_tensors[i].clone();
    }

    Ok(())
}
