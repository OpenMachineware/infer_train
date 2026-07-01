// src/ffi.rs
use std::os::raw::{c_void, c_float, c_int};

// ============================================================
// C 类型定义（对应 c_interface.h）
// ============================================================
#[repr(C)]
pub struct it_tensor {
    _private: [u8; 0],  // 不透明指针
}

unsafe impl Send for it_tensor {}
unsafe impl Sync for it_tensor {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum it_dtype_t {
    IT_DTYPE_F32 = 0,
    IT_DTYPE_F64 = 1,
    IT_DTYPE_F16 = 2,
    IT_DTYPE_BF16 = 3,
    IT_DTYPE_I8 = 4,
}

// ============================================================
// C 函数声明
// ============================================================
extern "C" {
    // Tensor 生命周期
    pub fn it_tensor_new(
        data: *const c_void,
        shape: *const usize,
        ndim: usize,
        dtype: it_dtype_t,
    ) -> *mut it_tensor;

    pub fn it_tensor_new_quantized(
        data: *const c_void,
        shape: *const usize,
        ndim: usize,
        dtype: it_dtype_t,
        scale: c_float,
        zero_point: c_float,
    ) -> *mut it_tensor;

    pub fn it_tensor_empty(
        shape: *const usize,
        ndim: usize,
        dtype: it_dtype_t,
    ) -> *mut it_tensor;

    pub fn it_tensor_empty_quantized(
        shape: *const usize,
        ndim: usize,
        dtype: it_dtype_t,
        scale: c_float,
        zero_point: c_float,
    ) -> *mut it_tensor;

    pub fn it_tensor_free(tensor: *mut it_tensor);

    // Tensor 属性
    pub fn it_tensor_ndim(tensor: *const it_tensor) -> usize;
    pub fn it_tensor_shape(tensor: *const it_tensor) -> *const usize;
    pub fn it_tensor_data(tensor: *const it_tensor) -> *const c_void;
    pub fn it_tensor_mutable_data(tensor: *mut it_tensor) -> *mut c_void;
    pub fn it_tensor_dtype(tensor: *const it_tensor) -> it_dtype_t;
    pub fn it_tensor_size(tensor: *const it_tensor) -> usize;
    pub fn it_tensor_elem_size(tensor: *const it_tensor) -> usize;
    pub fn it_tensor_scale(tensor: *const it_tensor) -> c_float;
    pub fn it_tensor_zero_point(tensor: *const it_tensor) -> c_float;
    pub fn it_tensor_set_quant_params(
        tensor: *mut it_tensor,
        scale: c_float,
        zero_point: c_float,
    );

    // 算子
    pub fn it_add(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_add_scalar(a: *const it_tensor, scalar: c_float) -> *mut it_tensor;
    pub fn it_add_n(tensors: *const *const it_tensor, n: usize) -> *mut it_tensor;
    pub fn it_matmul(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_batch_matmul(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_vec_matmul(vec: *const it_tensor, mat: *const it_tensor) -> *mut it_tensor;
    pub fn it_transpose(a: *const it_tensor) -> *mut it_tensor;

    pub fn it_relu(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_leaky_relu(input: *const it_tensor, alpha: c_float) -> *mut it_tensor;
    pub fn it_elu(input: *const it_tensor, alpha: c_float) -> *mut it_tensor;
    pub fn it_gelu(input: *const it_tensor) -> *mut it_tensor;

    pub fn it_softmax(input: *const it_tensor, dim: c_int) -> *mut it_tensor;
    pub fn it_log_softmax(input: *const it_tensor, dim: c_int) -> *mut it_tensor;
}

// ============================================================
// 安全封装：Tensor
// ============================================================
pub struct Tensor {
    ptr: *mut it_tensor,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    // ============================================================
    // FP64 支持
    // ============================================================
    pub fn new_f64(data: &[f64], shape: &[usize]) -> Self {
        let ptr = unsafe {
            it_tensor_new(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len(),
                it_dtype_t::IT_DTYPE_F64,
            )
        };
        Tensor { ptr }    }

    pub fn data_as_f64(&self) -> &[f64] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const f64 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    // ============================================================
    // FP32 创建浮点张量
    // ============================================================
    pub fn new_f32(data: &[f32], shape: &[usize]) -> Self {
        let ptr = unsafe {
            it_tensor_new(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len(),
                it_dtype_t::IT_DTYPE_F32,
            )
        };
        Tensor { ptr }
    }

    // 获取数据（只读）
    pub fn data_as_f32(&self) -> &[f32] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const f32 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    // 获取数据（可变）
    pub fn data_as_f32_mut(&mut self) -> &mut [f32] {
        let ptr = unsafe { it_tensor_mutable_data(self.ptr) as *mut f32 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts_mut(ptr, size) }
    }

    // ============================================================
    // FP16 支持（用 u16 存储）
    // ============================================================
    pub fn new_f16(data: &[u16], shape: &[usize]) -> Self {
        let ptr = unsafe {
            it_tensor_new(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len(),
                it_dtype_t::IT_DTYPE_F16,
            )
        };
        Tensor { ptr }
    }

    pub fn data_as_f16(&self) -> &[u16] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const u16 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    // ============================================================
    // BF16 支持（用 u16 存储）
    // ============================================================
    pub fn new_bf16(data: &[u16], shape: &[usize]) -> Self {
        let ptr = unsafe {
            it_tensor_new(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len(),
                it_dtype_t::IT_DTYPE_BF16,
            )
        };
        Tensor { ptr }
    }

    pub fn data_as_bf16(&self) -> &[u16] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const u16 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    // ============================================================
    // INT8 支持
    // ============================================================
    pub fn new_i8(data: &[i8], shape: &[usize], scale: f32, zero_point: f32) -> Self {
        let ptr = unsafe {
            it_tensor_new_quantized(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len(),
                it_dtype_t::IT_DTYPE_I8,
                scale,
                zero_point,
            )
        };
        Tensor { ptr }
    }

    pub fn data_as_i8(&self) -> &[i8] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const i8 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    // ============================================================
    // 通用：从 Python 传入的 f32 自动转换
    // ============================================================
    pub fn from_python_f32(data: Vec<f32>, shape: Vec<usize>, dtype: it_dtype_t) -> Self {
        match dtype {
            it_dtype_t::IT_DTYPE_F32 => Tensor::new_f32(&data, &shape),
            it_dtype_t::IT_DTYPE_F64 => {
                // f64 支持
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let ptr = unsafe {
                    it_tensor_new(
                        data_f64.as_ptr() as *const c_void,
                        shape.as_ptr(),
                        shape.len(),
                        it_dtype_t::IT_DTYPE_F64,
                    )
                };
                Tensor { ptr }
            }
            _ => Tensor::new_f32(&data, &shape),
        }
    }

    // 创建空张量
    pub fn empty_f32(shape: &[usize]) -> Self {
        let ptr = unsafe {
            it_tensor_empty(
                shape.as_ptr(),
                shape.len(),
                it_dtype_t::IT_DTYPE_F32,
            )
        };
        Tensor { ptr }
    }

    // 从 C 指针创建（不转移所有权）
    pub unsafe fn from_ptr(ptr: *mut it_tensor) -> Self {
        Tensor { ptr }
    }

    pub fn shape(&self) -> Vec<usize> {
        let ptr = unsafe { it_tensor_shape(self.ptr) };
        let ndim = unsafe { it_tensor_ndim(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, ndim).to_vec() }
    }

    pub fn dtype(&self) -> it_dtype_t {
        unsafe { it_tensor_dtype(self.ptr) }
    }

    pub fn scale(&self) -> f32 {
        unsafe { it_tensor_scale(self.ptr) }
    }

    pub fn zero_point(&self) -> f32 {
        unsafe { it_tensor_zero_point(self.ptr) }
    }

    pub fn set_quant_params(&mut self, scale: f32, zero_point: f32) {
        unsafe { it_tensor_set_quant_params(self.ptr, scale, zero_point) }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.dtype(), it_dtype_t::IT_DTYPE_I8)
    }

    // ============================================================
    // 公开数据指针（供 lib.rs 使用）
    // ============================================================
    pub fn size(&self) -> usize {
        unsafe { it_tensor_size(self.ptr) }
    }

    pub fn data_ptr_f64(&self) -> *const f64 {
        unsafe { it_tensor_data(self.ptr) as *const f64 }
    }

    pub fn data_ptr_f16(&self) -> *const u16 {
        unsafe { it_tensor_data(self.ptr) as *const u16 }
    }

    pub fn data_ptr_bf16(&self) -> *const u16 {
        unsafe { it_tensor_data(self.ptr) as *const u16 }
    }

    pub fn data_ptr_i8(&self) -> *const i8 {
        unsafe { it_tensor_data(self.ptr) as *const i8 }
    }

    // ============================================================
    // 算子
    // ============================================================
    pub fn add(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_add(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_matmul(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn relu(&self) -> Tensor {
        let ptr = unsafe { it_relu(self.ptr) };
        Tensor { ptr }
    }

    pub fn softmax(&self, dim: i32) -> Tensor {
        let ptr = unsafe { it_softmax(self.ptr, dim) };
        Tensor { ptr }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { it_tensor_free(self.ptr) };
        }
    }
}

// ============================================================
// 测试
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let t = Tensor::new_f32(&data, &shape);
        assert_eq!(t.shape(), vec![2, 2]);
        assert_eq!(t.data_as_f32(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(!t.is_quantized());
    }

    #[test]
    fn test_add_f32() {
        let a = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::new_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.add(&b);
        assert_eq!(c.data_as_f32(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul_f32() {
        let a = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::new_f32(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data_as_f32(), &[4.0, 6.0, 10.0, 12.0]);
    }

    #[test]
    fn test_relu_f32() {
        let input = Tensor::new_f32(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
        let output = input.relu();
        assert_eq!(output.data_as_f32(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_quantized_tensor() {
        let data = vec![1, -2, 3, -4];
        let t = Tensor::new_i8(&data, &[2, 2], 0.01, 0.0);
        assert!(t.is_quantized());
        assert_eq!(t.scale(), 0.01);
        assert_eq!(t.zero_point(), 0.0);
    }
}
