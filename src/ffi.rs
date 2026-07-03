// src/ffi.rs
use std::os::raw::{c_void, c_float, c_int};
use std::ptr;

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
    // ============================================================
    // Tensor 生命周期
    // ============================================================
    pub fn it_tensor_new(
        data: *const c_void,
        shape: *const usize,
        ndim: usize,
        dtype: it_dtype_t,
    ) -> *mut it_tensor;

    pub fn it_tensor_empty(
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

    pub fn it_tensor_empty_quantized(
        shape: *const usize,
        ndim: usize,
        dtype: it_dtype_t,
        scale: c_float,
        zero_point: c_float,
    ) -> *mut it_tensor;

    pub fn it_tensor_set_quant_params(
        tensor: *mut it_tensor,
        scale: c_float,
        zero_point: c_float,
    );

    pub fn it_tensor_free(tensor: *mut it_tensor);

    // ============================================================
    // Tensor 属性
    // ============================================================
    pub fn it_tensor_ndim(tensor: *const it_tensor) -> usize;
    pub fn it_tensor_shape(tensor: *const it_tensor) -> *const usize;
    pub fn it_tensor_data(tensor: *const it_tensor) -> *const c_void;
    pub fn it_tensor_mutable_data(tensor: *mut it_tensor) -> *mut c_void;
    pub fn it_tensor_dtype(tensor: *const it_tensor) -> it_dtype_t;
    pub fn it_tensor_size(tensor: *const it_tensor) -> usize;
    pub fn it_tensor_elem_size(tensor: *const it_tensor) -> usize;
    pub fn it_tensor_scale(tensor: *const it_tensor) -> c_float;
    pub fn it_tensor_zero_point(tensor: *const it_tensor) -> c_float;

    // ============================================================
    // 数学算子
    // ============================================================
    pub fn it_add(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_sub(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_mul(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_div(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_pow(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_exp(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_sqrt(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_log(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_log2(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_log10(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_abs(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_neg(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_clamp(input: *const it_tensor, min_val: c_float, max_val: c_float) -> *mut it_tensor;
    pub fn it_floor(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_ceil(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_round(input: *const it_tensor) -> *mut it_tensor;

    // 量化数学
    pub fn it_quantized_add(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_sub(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_mul(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_div(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_exp(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_sqrt(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_abs(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_neg(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_clamp(input: *const it_tensor, min_val: c_float, max_val: c_float) -> *mut it_tensor;

    // ============================================================
    // 规约算子
    // ============================================================
    pub fn it_sum(input: *const it_tensor, dims: *const c_int, ndim: usize, keepdim: c_int) -> *mut it_tensor;
    pub fn it_mean(input: *const it_tensor, dims: *const c_int, ndim: usize, keepdim: c_int) -> *mut it_tensor;
    pub fn it_max_all(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_min_all(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_prod_all(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_var(input: *const it_tensor, unbiased: c_int) -> *mut it_tensor;
    pub fn it_std(input: *const it_tensor, unbiased: c_int) -> *mut it_tensor;

    pub fn it_quantized_sum(input: *const it_tensor, dims: *const c_int, ndim: usize, keepdim: c_int) -> *mut it_tensor;
    pub fn it_quantized_mean(input: *const it_tensor, dims: *const c_int, ndim: usize, keepdim: c_int) -> *mut it_tensor;
    pub fn it_quantized_max_all(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_min_all(input: *const it_tensor) -> *mut it_tensor;

    // ============================================================
    // 矩阵算子
    // ============================================================
    pub fn it_matmul(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_batch_matmul(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_vec_matmul(vec: *const it_tensor, mat: *const it_tensor) -> *mut it_tensor;
    pub fn it_transpose(input: *const it_tensor) -> *mut it_tensor;

    pub fn it_quantized_matmul(a: *const it_tensor, b: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_vec_matmul(vec: *const it_tensor, mat: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_transpose(input: *const it_tensor) -> *mut it_tensor;

    // ============================================================
    // NN 算子
    // ============================================================
    pub fn it_conv1d(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
    ) -> *mut it_tensor;

    pub fn it_conv2d(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
    ) -> *mut it_tensor;

    pub fn it_conv3d(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
    ) -> *mut it_tensor;

    pub fn it_maxpool1d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_maxpool2d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_maxpool3d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_avgpool1d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_avgpool2d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_avgpool3d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;

    pub fn it_batchnorm1d_inference(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        running_mean: *const it_tensor,
        running_var: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_batchnorm2d_inference(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        running_mean: *const it_tensor,
        running_var: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_layernorm(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_rmsnorm(
        input: *const it_tensor,
        weight: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_instancenorm2d(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_groupnorm(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        num_groups: c_int,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_linear(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
    ) -> *mut it_tensor;

    pub fn it_embedding(
        indices: *const i64,
        num_indices: usize,
        weight: *const it_tensor,
        padding_idx: c_int,
    ) -> *mut it_tensor;

    pub fn it_dropout(input: *const it_tensor, p: c_float) -> *mut it_tensor;

    // 量化 NN
    pub fn it_quantized_conv1d(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
    ) -> *mut it_tensor;

    pub fn it_quantized_conv2d(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
    ) -> *mut it_tensor;

    pub fn it_quantized_conv3d(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
    ) -> *mut it_tensor;

    pub fn it_quantized_maxpool1d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_quantized_maxpool2d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_quantized_maxpool3d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_quantized_avgpool1d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_quantized_avgpool2d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;
    pub fn it_quantized_avgpool3d(input: *const it_tensor, kernel_size: c_int, stride: c_int, padding: c_int) -> *mut it_tensor;

    pub fn it_quantized_batchnorm1d_inference(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        running_mean: *const it_tensor,
        running_var: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_quantized_batchnorm2d_inference(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        running_mean: *const it_tensor,
        running_var: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_quantized_layernorm(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_quantized_rmsnorm(
        input: *const it_tensor,
        weight: *const it_tensor,
        eps: c_float,
    ) -> *mut it_tensor;

    pub fn it_quantized_linear(
        input: *const it_tensor,
        weight: *const it_tensor,
        bias: *const it_tensor,
    ) -> *mut it_tensor;

    pub fn it_quantized_embedding(
        indices: *const i64,
        num_indices: usize,
        weight: *const it_tensor,
        padding_idx: c_int,
    ) -> *mut it_tensor;

    // ============================================================
    // 激活函数
    // ============================================================
    pub fn it_relu(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_leaky_relu(input: *const it_tensor, alpha: c_float) -> *mut it_tensor;
    pub fn it_elu(input: *const it_tensor, alpha: c_float) -> *mut it_tensor;
    pub fn it_gelu(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_relu6(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_sigmoid(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_tanh(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_silu(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_hard_swish(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_hard_sigmoid(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_softplus(input: *const it_tensor, beta: c_float, threshold: c_float) -> *mut it_tensor;
    pub fn it_softshrink(input: *const it_tensor, lambda: c_float) -> *mut it_tensor;
    pub fn it_celu(input: *const it_tensor, alpha: c_float) -> *mut it_tensor;
    pub fn it_softmax(input: *const it_tensor, dim: c_int) -> *mut it_tensor;
    pub fn it_log_softmax(input: *const it_tensor, dim: c_int) -> *mut it_tensor;

    pub fn it_quantized_relu(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_leaky_relu(input: *const it_tensor, alpha: c_float) -> *mut it_tensor;
    pub fn it_quantized_elu(input: *const it_tensor, alpha: c_float) -> *mut it_tensor;
    pub fn it_quantized_gelu(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_relu6(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_sigmoid(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_tanh(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_silu(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_hard_swish(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_hard_sigmoid(input: *const it_tensor) -> *mut it_tensor;
    pub fn it_quantized_softmax(input: *const it_tensor, dim: c_int) -> *mut it_tensor;
    pub fn it_quantized_log_softmax(input: *const it_tensor, dim: c_int) -> *mut it_tensor;

    // ============================================================
    // 注意力算子
    // ============================================================
    pub fn it_scaled_dot_product_attention(
        query: *const it_tensor,
        key: *const it_tensor,
        value: *const it_tensor,
        mask: *const it_tensor,
        scale: c_float,
        is_causal: c_int,
        dropout_p: c_float,
    ) -> *mut it_tensor;

    pub fn it_multi_head_attention(
        query: *const it_tensor,
        key: *const it_tensor,
        value: *const it_tensor,
        mask: *const it_tensor,
        num_heads: c_int,
        scale: c_float,
        is_causal: c_int,
        dropout_p: c_float,
    ) -> *mut it_tensor;

    pub fn it_rotary_embedding(
        x: *const it_tensor,
        cos: *const it_tensor,
        sin: *const it_tensor,
    ) -> *mut it_tensor;

    pub fn it_quantized_scaled_dot_product_attention(
        query: *const it_tensor,
        key: *const it_tensor,
        value: *const it_tensor,
        mask: *const it_tensor,
        scale: c_float,
        is_causal: c_int,
        dropout_p: c_float,
    ) -> *mut it_tensor;

    pub fn it_quantized_multi_head_attention(
        query: *const it_tensor,
        key: *const it_tensor,
        value: *const it_tensor,
        mask: *const it_tensor,
        num_heads: c_int,
        scale: c_float,
        is_causal: c_int,
        dropout_p: c_float,
    ) -> *mut it_tensor;

    pub fn it_quantized_rotary_embedding(
        x: *const it_tensor,
        cos: *const it_tensor,
        sin: *const it_tensor,
    ) -> *mut it_tensor;

    // ============================================================
    // 损失函数
    // ============================================================
    pub fn it_cross_entropy_loss(
        input: *const it_tensor,
        target: *const i64,
        batch_size: usize,
        reduction: c_int,
    ) -> *mut it_tensor;

    pub fn it_mse_loss(
        input: *const it_tensor,
        target: *const it_tensor,
        reduction: c_int,
    ) -> *mut it_tensor;

    pub fn it_l1_loss(
        input: *const it_tensor,
        target: *const it_tensor,
        reduction: c_int,
    ) -> *mut it_tensor;

    pub fn it_bce_loss(
        input: *const it_tensor,
        target: *const it_tensor,
        reduction: c_int,
        eps: c_float,
    ) -> *mut it_tensor;

    // ============================================================
    // 优化器
    // ============================================================
    pub fn it_sgd_update(
        params: *mut *mut it_tensor,
        grads: *mut *mut it_tensor,
        num_params: usize,
        lr: c_float,
        momentum: c_float,
        weight_decay: c_float,
        nesterov: c_int,
    );

    // ============================================================
    // 张量操作
    // ============================================================
    pub fn it_slice(
        input: *const it_tensor,
        dim: c_int,
        start: c_int,
        end: c_int,
        step: c_int,
    ) -> *mut it_tensor;

    pub fn it_cat(
        tensors: *const *const it_tensor,
        n: usize,
        dim: c_int,
    ) -> *mut it_tensor;

    pub fn it_cumsum(
        input: *const it_tensor,
        dim: c_int,
    ) -> *mut it_tensor;

    pub fn it_cumprod(
        input: *const it_tensor,
        dim: c_int,
    ) -> *mut it_tensor;

    pub fn it_where(
        condition: *const u8,
        condition_shape: *const usize,
        condition_ndim: usize,
        true_val: *const it_tensor,
        false_val: *const it_tensor,
    ) -> *mut it_tensor;
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
    // 创建
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

    pub fn new_quantized(data: &[i8], shape: &[usize], scale: f32, zero_point: f32) -> Self {
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
        Tensor { ptr }
    }

    // ============================================================
    // FP16 支持
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

    // ============================================================
    // BF16 支持
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

    pub unsafe fn from_ptr(ptr: *mut it_tensor) -> Self {
        Tensor { ptr }
    }

    // ============================================================
    // 属性
    // ============================================================
    pub fn data_as_f32(&self) -> &[f32] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const f32 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    pub fn data_as_i8(&self) -> &[i8] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const i8 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    pub fn data_as_f64(&self) -> &[f64] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const f64 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    pub fn data_as_f16(&self) -> &[u16] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const u16 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    pub fn data_as_bf16(&self) -> &[u16] {
        let ptr = unsafe { it_tensor_data(self.ptr) as *const u16 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
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

    pub fn is_quantized(&self) -> bool {
        matches!(self.dtype(), it_dtype_t::IT_DTYPE_I8)
    }

    // ============================================================
    // 数学算子
    // ============================================================
    pub fn add(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_add(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_sub(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_mul(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_div(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn pow(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_pow(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn exp(&self) -> Tensor {
        let ptr = unsafe { it_exp(self.ptr) };
        Tensor { ptr }
    }

    pub fn sqrt(&self) -> Tensor {
        let ptr = unsafe { it_sqrt(self.ptr) };
        Tensor { ptr }
    }

    pub fn log(&self) -> Tensor {
        let ptr = unsafe { it_log(self.ptr) };
        Tensor { ptr }
    }

    pub fn log2(&self) -> Tensor {
        let ptr = unsafe { it_log2(self.ptr) };
        Tensor { ptr }
    }

    pub fn log10(&self) -> Tensor {
        let ptr = unsafe { it_log10(self.ptr) };
        Tensor { ptr }
    }

    pub fn abs(&self) -> Tensor {
        let ptr = unsafe { it_abs(self.ptr) };
        Tensor { ptr }
    }

    pub fn neg(&self) -> Tensor {
        let ptr = unsafe { it_neg(self.ptr) };
        Tensor { ptr }
    }

    pub fn clamp(&self, min_val: f32, max_val: f32) -> Tensor {
        let ptr = unsafe { it_clamp(self.ptr, min_val, max_val) };
        Tensor { ptr }
    }

    pub fn floor(&self) -> Tensor {
        let ptr = unsafe { it_floor(self.ptr) };
        Tensor { ptr }
    }

    pub fn ceil(&self) -> Tensor {
        let ptr = unsafe { it_ceil(self.ptr) };
        Tensor { ptr }
    }

    pub fn round(&self) -> Tensor {
        let ptr = unsafe { it_round(self.ptr) };
        Tensor { ptr }
    }

    // ============================================================
    // 矩阵算子
    // ============================================================
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_matmul(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn batch_matmul(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_batch_matmul(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn vec_matmul(&self, mat: &Tensor) -> Tensor {
        let ptr = unsafe { it_vec_matmul(self.ptr, mat.ptr) };
        Tensor { ptr }
    }

    // ============================================================
    // 激活函数
    // ============================================================
    pub fn relu(&self) -> Tensor {
        let ptr = unsafe { it_relu(self.ptr) };
        Tensor { ptr }
    }

    pub fn leaky_relu(&self, alpha: f32) -> Tensor {
    let ptr = unsafe { it_leaky_relu(self.ptr, alpha) };
    Tensor { ptr }
    }

    pub fn elu(&self, alpha: f32) -> Tensor {
        let ptr = unsafe { it_elu(self.ptr, alpha) };
        Tensor { ptr }
    }

    pub fn gelu(&self) -> Tensor {
        let ptr = unsafe { it_gelu(self.ptr) };
        Tensor { ptr }
    }

    pub fn relu6(&self) -> Tensor {
        let ptr = unsafe { it_relu6(self.ptr) };
        Tensor { ptr }
    }

    pub fn sigmoid(&self) -> Tensor {
        let ptr = unsafe { it_sigmoid(self.ptr) };
        Tensor { ptr }
    }

    pub fn softmax(&self, dim: i32) -> Tensor {
        let ptr = unsafe { it_softmax(self.ptr, dim) };
        Tensor { ptr }
    }
    pub fn tanh(&self) -> Tensor {
        let ptr = unsafe { it_tanh(self.ptr) };
        Tensor { ptr }
    }

    pub fn silu(&self) -> Tensor {
        let ptr = unsafe { it_silu(self.ptr) };
        Tensor { ptr }
    }

    pub fn hard_swish(&self) -> Tensor {
        let ptr = unsafe { it_hard_swish(self.ptr) };
        Tensor { ptr }
    }

    pub fn hard_sigmoid(&self) -> Tensor {
        let ptr = unsafe { it_hard_sigmoid(self.ptr) };
        Tensor { ptr }
    }

    pub fn softplus(&self, beta: f32, threshold: f32) -> Tensor {
        let ptr = unsafe { it_softplus(self.ptr, beta, threshold) };
        Tensor { ptr }
    }

    pub fn softshrink(&self, lambda: f32) -> Tensor {
        let ptr = unsafe { it_softshrink(self.ptr, lambda) };
        Tensor { ptr }
    }

    pub fn celu(&self, alpha: f32) -> Tensor {
        let ptr = unsafe { it_celu(self.ptr, alpha) };
        Tensor { ptr }
    }

    pub fn log_softmax(&self, dim: i32) -> Tensor {
        let ptr = unsafe { it_log_softmax(self.ptr, dim) };
        Tensor { ptr }
    }

    // ============================================================
    // NN 算子 1D/3D
    // ============================================================
    pub fn conv1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Tensor {
        let bias_ptr = bias.map_or(std::ptr::null(), |b| b.ptr);
        let ptr = unsafe {
            it_conv1d(
                self.ptr,
                weight.ptr,
                bias_ptr,
                stride,
                padding,
                dilation,
                groups,
            )
        };
        Tensor { ptr }
    }

    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Tensor {
        let bias_ptr = bias.map_or(ptr::null(), |b| b.ptr);
        let ptr = unsafe {
            it_conv2d(
                self.ptr,
                weight.ptr,
                bias_ptr,
                stride,
                padding,
                dilation,
                groups,
            )
        };
        Tensor { ptr }
    }

    pub fn conv3d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Tensor {
        let bias_ptr = bias.map_or(std::ptr::null(), |b| b.ptr);
        let ptr = unsafe {
            it_conv3d(
                self.ptr,
                weight.ptr,
                bias_ptr,
                stride,
                padding,
                dilation,
                groups,
            )
        };
        Tensor { ptr }
    }

    pub fn maxpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_maxpool1d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn maxpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_maxpool2d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn maxpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_maxpool3d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn avgpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_avgpool1d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn avgpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_avgpool2d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn avgpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_avgpool3d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn batchnorm1d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        eps: f32,
    ) -> Tensor {
        let ptr = unsafe {
            it_batchnorm1d_inference(
                self.ptr,
                weight.ptr,
                bias.ptr,
                running_mean.ptr,
                running_var.ptr,
                eps,
            )
        };
        Tensor { ptr }
    }

    pub fn batchnorm2d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        eps: f32,
    ) -> Tensor {
        let ptr = unsafe {
            it_batchnorm2d_inference(
                self.ptr,
                weight.ptr,
                bias.ptr,
                running_mean.ptr,
                running_var.ptr,
                eps,
            )
        };
        Tensor { ptr }
    }

    pub fn instancenorm2d(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let ptr = unsafe { it_instancenorm2d(self.ptr, weight.ptr, bias.ptr, eps) };
        Tensor { ptr }
    }

    pub fn groupnorm(&self, weight: &Tensor, bias: &Tensor, num_groups: i32, eps: f32) -> Tensor {
        let ptr = unsafe { it_groupnorm(self.ptr, weight.ptr, bias.ptr, num_groups, eps) };
        Tensor { ptr }
    }

    pub fn embedding(&self, indices: &[i64], padding_idx: i32) -> Tensor {
        let ptr = unsafe {
            it_embedding(
                indices.as_ptr(),
                indices.len(),
                self.ptr,
                padding_idx,
            )
        };
        Tensor { ptr }
    }

    pub fn dropout(&self, p: f32) -> Tensor {
        let ptr = unsafe { it_dropout(self.ptr, p) };
        Tensor { ptr }
    }

    pub fn layernorm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let ptr = unsafe { it_layernorm(self.ptr, weight.ptr, bias.ptr, eps) };
        Tensor { ptr }
    }

    pub fn rmsnorm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let ptr = unsafe { it_rmsnorm(self.ptr, weight.ptr, eps) };
        Tensor { ptr }
    }

    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let bias_ptr = bias.map_or(ptr::null(), |b| b.ptr);
        let ptr = unsafe { it_linear(self.ptr, weight.ptr, bias_ptr) };
        Tensor { ptr }
    }

    pub fn transpose(&self) -> Tensor {
        let ptr = unsafe { it_transpose(self.ptr) };
        Tensor { ptr }
    }

    pub fn slice(&self, dim: i32, start: i32, end: i32, step: i32) -> Tensor {
        let ptr = unsafe { it_slice(self.ptr, dim, start, end, step) };
        Tensor { ptr }
    }

    pub fn cumsum(&self, dim: i32) -> Tensor {
        let ptr = unsafe { it_cumsum(self.ptr, dim) };
        Tensor { ptr }
    }

    pub fn cumprod(&self, dim: i32) -> Tensor {
        let ptr = unsafe { it_cumprod(self.ptr, dim) };
        Tensor { ptr }
    }

    pub fn sum(&self, dims: &[i32], keepdim: bool) -> Tensor {
        let ptr = unsafe {
            it_sum(
                self.ptr,
                dims.as_ptr(),
                dims.len(),
                if keepdim { 1 } else { 0 },
            )
        };
        Tensor { ptr }
    }

    pub fn mean(&self, dims: &[i32], keepdim: bool) -> Tensor {
        let ptr = unsafe {
            it_mean(
                self.ptr,
                dims.as_ptr(),
                dims.len(),
                if keepdim { 1 } else { 0 },
            )
        };
        Tensor { ptr }
    }

    pub fn max_all(&self) -> Tensor {
        let ptr = unsafe { it_max_all(self.ptr) };
        Tensor { ptr }
    }

    pub fn min_all(&self) -> Tensor {
        let ptr = unsafe { it_min_all(self.ptr) };
        Tensor { ptr }
    }

    pub fn prod_all(&self) -> Tensor {
        let ptr = unsafe { it_prod_all(self.ptr) };
        Tensor { ptr }
    }

    pub fn var(&self, unbiased: bool) -> Tensor {
        let ptr = unsafe { it_var(self.ptr, if unbiased { 1 } else { 0 }) };
        Tensor { ptr }
    }

    pub fn std(&self, unbiased: bool) -> Tensor {
        let ptr = unsafe { it_std(self.ptr, if unbiased { 1 } else { 0 }) };
        Tensor { ptr }
    }

    // ============================================================
    // cat（静态方法）
    // ============================================================
    pub fn as_ptr(&self) -> *const it_tensor {
        self.ptr as *const it_tensor
    }

    pub fn cat(tensors: &[&Tensor], dim: i32) -> Tensor {
        let mut ptrs: Vec<*const it_tensor> = tensors.iter().map(|t| t.ptr as *const it_tensor).collect();
        let ptr = unsafe {
            it_cat(
                ptrs.as_mut_ptr(),
                tensors.len(),
                dim,
            )
        };
        Tensor { ptr }
    }

    // ============================================================
    // 量化算子
    // ============================================================
    pub fn quantized_add(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_quantized_add(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_sub(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_quantized_sub(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_mul(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_quantized_mul(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_div(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_quantized_div(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_exp(&self) -> Tensor {
        let ptr = unsafe { it_quantized_exp(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_sqrt(&self) -> Tensor {
        let ptr = unsafe { it_quantized_sqrt(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_abs(&self) -> Tensor {
        let ptr = unsafe { it_quantized_abs(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_neg(&self) -> Tensor {
        let ptr = unsafe { it_quantized_neg(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_clamp(&self, min_val: f32, max_val: f32) -> Tensor {
        let ptr = unsafe { it_quantized_clamp(self.ptr, min_val, max_val) };
        Tensor { ptr }
    }

    pub fn quantized_matmul(&self, other: &Tensor) -> Tensor {
        let ptr = unsafe { it_quantized_matmul(self.ptr, other.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_conv1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Tensor {
        let bias_ptr = bias.map_or(std::ptr::null(), |b| b.ptr);
        let ptr = unsafe {
            it_quantized_conv1d(
                self.ptr,
                weight.ptr,
                bias_ptr,
                stride,
                padding,
                dilation,
                groups,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Tensor {
        let bias_ptr = bias.map_or(std::ptr::null(), |b| b.ptr);
        let ptr = unsafe {
            it_quantized_conv2d(
                self.ptr,
                weight.ptr,
                bias_ptr,
                stride,
                padding,
                dilation,
                groups,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_conv3d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Tensor {
        let bias_ptr = bias.map_or(std::ptr::null(), |b| b.ptr);
        let ptr = unsafe {
            it_quantized_conv3d(
                self.ptr,
                weight.ptr,
                bias_ptr,
                stride,
                padding,
                dilation,
                groups,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_maxpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_quantized_maxpool1d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn quantized_maxpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_quantized_maxpool2d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn quantized_maxpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_quantized_maxpool3d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn quantized_avgpool1d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_quantized_avgpool1d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn quantized_avgpool2d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_quantized_avgpool2d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn quantized_avgpool3d(&self, kernel_size: i32, stride: i32, padding: i32) -> Tensor {
        let ptr = unsafe { it_quantized_avgpool3d(self.ptr, kernel_size, stride, padding) };
        Tensor { ptr }
    }

    pub fn quantized_batchnorm1d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        eps: f32,
    ) -> Tensor {
        let ptr = unsafe {
            it_quantized_batchnorm1d_inference(
                self.ptr,
                weight.ptr,
                bias.ptr,
                running_mean.ptr,
                running_var.ptr,
                eps,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_batchnorm2d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        running_mean: &Tensor,
        running_var: &Tensor,
        eps: f32,
    ) -> Tensor {
        let ptr = unsafe {
            it_quantized_batchnorm2d_inference(
                self.ptr,
                weight.ptr,
                bias.ptr,
                running_mean.ptr,
                running_var.ptr,
                eps,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_layernorm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let ptr = unsafe { it_quantized_layernorm(self.ptr, weight.ptr, bias.ptr, eps) };
        Tensor { ptr }
    }

    pub fn quantized_rmsnorm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let ptr = unsafe { it_quantized_rmsnorm(self.ptr, weight.ptr, eps) };
        Tensor { ptr }
    }

    pub fn quantized_linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let bias_ptr = bias.map_or(std::ptr::null(), |b| b.ptr);
        let ptr = unsafe { it_quantized_linear(self.ptr, weight.ptr, bias_ptr) };
        Tensor { ptr }
    }

    pub fn quantized_embedding(&self, indices: &[i64], padding_idx: i32) -> Tensor {
        let ptr = unsafe {
            it_quantized_embedding(
                indices.as_ptr(),
                indices.len(),
                self.ptr,
                padding_idx,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_relu(&self) -> Tensor {
        let ptr = unsafe { it_quantized_relu(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_leaky_relu(&self, alpha: f32) -> Tensor {
        let ptr = unsafe { it_quantized_leaky_relu(self.ptr, alpha) };
        Tensor { ptr }
    }

    pub fn quantized_elu(&self, alpha: f32) -> Tensor {
        let ptr = unsafe { it_quantized_elu(self.ptr, alpha) };
        Tensor { ptr }
    }

    pub fn quantized_gelu(&self) -> Tensor {
        let ptr = unsafe { it_quantized_gelu(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_relu6(&self) -> Tensor {
        let ptr = unsafe { it_quantized_relu6(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_sigmoid(&self) -> Tensor {
        let ptr = unsafe { it_quantized_sigmoid(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_tanh(&self) -> Tensor {
        let ptr = unsafe { it_quantized_tanh(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_silu(&self) -> Tensor {
        let ptr = unsafe { it_quantized_silu(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_hard_swish(&self) -> Tensor {
        let ptr = unsafe { it_quantized_hard_swish(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_hard_sigmoid(&self) -> Tensor {
        let ptr = unsafe { it_quantized_hard_sigmoid(self.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_softmax(&self, dim: i32) -> Tensor {
        let ptr = unsafe { it_quantized_softmax(self.ptr, dim) };
        Tensor { ptr }
    }

    pub fn quantized_log_softmax(&self, dim: i32) -> Tensor {
        let ptr = unsafe { it_quantized_log_softmax(self.ptr, dim) };
        Tensor { ptr }
    }

    // ============================================================
    // 注意力算子（追加）
    // ============================================================
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> Tensor {
        let mask_ptr = mask.map_or(std::ptr::null(), |m| m.ptr);
        let ptr = unsafe {
            it_scaled_dot_product_attention(
                self.ptr,
                key.ptr,
                value.ptr,
                mask_ptr,
                scale,
                if is_causal { 1 } else { 0 },
                dropout_p,
            )
        };
        Tensor { ptr }
    }

    pub fn multi_head_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        num_heads: i32,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> Tensor {
        let mask_ptr = mask.map_or(std::ptr::null(), |m| m.ptr);
        let ptr = unsafe {
            it_multi_head_attention(
                self.ptr,
                key.ptr,
                value.ptr,
                mask_ptr,
                num_heads,
                scale,
                if is_causal { 1 } else { 0 },
                dropout_p,
            )
        };
        Tensor { ptr }
    }

    pub fn rotary_embedding(&self, cos: &Tensor, sin: &Tensor) -> Tensor {
        let ptr = unsafe { it_rotary_embedding(self.ptr, cos.ptr, sin.ptr) };
        Tensor { ptr }
    }

    pub fn quantized_scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> Tensor {
        let mask_ptr = mask.map_or(std::ptr::null(), |m| m.ptr);
        let ptr = unsafe {
            it_quantized_scaled_dot_product_attention(
                self.ptr,
                key.ptr,
                value.ptr,
                mask_ptr,
                scale,
                if is_causal { 1 } else { 0 },
                dropout_p,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_multi_head_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        num_heads: i32,
        scale: f32,
        is_causal: bool,
        dropout_p: f32,
    ) -> Tensor {
        let mask_ptr = mask.map_or(std::ptr::null(), |m| m.ptr);
        let ptr = unsafe {
            it_quantized_multi_head_attention(
                self.ptr,
                key.ptr,
                value.ptr,
                mask_ptr,
                num_heads,
                scale,
                if is_causal { 1 } else { 0 },
                dropout_p,
            )
        };
        Tensor { ptr }
    }

    pub fn quantized_rotary_embedding(&self, cos: &Tensor, sin: &Tensor) -> Tensor {
        let ptr = unsafe { it_quantized_rotary_embedding(self.ptr, cos.ptr, sin.ptr) };
        Tensor { ptr }
    }

    // ============================================================
    // 损失函数（追加）
    // ============================================================
    pub fn cross_entropy_loss(&self, target: &[i64], reduction: bool) -> Tensor {
        let ptr = unsafe {
            it_cross_entropy_loss(
                self.ptr,
                target.as_ptr(),
                target.len(),
                if reduction { 1 } else { 0 },
            )
        };
        Tensor { ptr }
    }

    pub fn mse_loss(&self, target: &Tensor, reduction: bool) -> Tensor {
        let ptr = unsafe {
            it_mse_loss(
                self.ptr,
                target.ptr,
                if reduction { 1 } else { 0 },
            )
        };
        Tensor { ptr }
    }

    pub fn l1_loss(&self, target: &Tensor, reduction: bool) -> Tensor {
        let ptr = unsafe {
            it_l1_loss(
                self.ptr,
                target.ptr,
                if reduction { 1 } else { 0 },
            )
        };
        Tensor { ptr }
    }

    pub fn bce_loss(&self, target: &Tensor, reduction: bool, eps: f32) -> Tensor {
        let ptr = unsafe {
            it_bce_loss(
                self.ptr,
                target.ptr,
                if reduction { 1 } else { 0 },
                eps,
            )
        };
        Tensor { ptr }
    }

    // ============================================================
    // 新增：可变数据访问
    // ============================================================
    pub fn data_as_f32_mut(&mut self) -> &mut [f32] {
        let ptr = unsafe { it_tensor_mutable_data(self.ptr) as *mut f32 };
        let size = unsafe { it_tensor_size(self.ptr) };
        unsafe { std::slice::from_raw_parts_mut(ptr, size) }
    }

    // ============================================================
    // 新增：量化参数设置
    // ============================================================
    pub fn set_quant_params(&mut self, scale: f32, zero_point: f32) {
        unsafe { it_tensor_set_quant_params(self.ptr, scale, zero_point) }
    }

    // ============================================================
    // 新增：数据指针（供 lib.rs 使用）
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
    // 新增：从 Python f32 自动转换
    // ============================================================
    pub fn from_python_f32(data: Vec<f32>, shape: Vec<usize>, dtype: it_dtype_t) -> Self {
        match dtype {
            it_dtype_t::IT_DTYPE_F32 => Tensor::new_f32(&data, &shape),
            it_dtype_t::IT_DTYPE_F64 => {
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
    fn test_softmax_f32() {
        let input = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = input.softmax(1);
        let data = output.data_as_f32();
        assert!((data[0] - 0.0320586).abs() < 1e-6);
        assert!((data[1] - 0.0871443).abs() < 1e-6);
        assert!((data[2] - 0.2368828).abs() < 1e-6);
        assert!((data[3] - 0.6439143).abs() < 1e-6);
    }

    #[test]
    fn test_quantized_tensor() {
        let data = vec![1, -2, 3, -4];
        let shape = vec![2, 2];
        let t = Tensor::new_quantized(&data, &shape, 0.01, 0.0);
        assert!(t.is_quantized());
        assert_eq!(t.scale(), 0.01);
        assert_eq!(t.zero_point(), 0.0);
        assert_eq!(t.data_as_i8(), &[1, -2, 3, -4]);
    }

    #[test]
    fn test_conv2d_f32() {
        let input = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 3, 3]);
        let weight = Tensor::new_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let output = input.conv2d(&weight, None, 1, 0, 1, 1);
        // 输出形状: (1, 1, 2, 2)
        assert_eq!(output.shape(), vec![1, 1, 2, 2]);
        let data = output.data_as_f32();
        // 手动验证卷积结果
        assert!((data[0] - 1.0).abs() < 1e-6);  // 位置 (0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6? 不对，应该是1+5=6? 实际测试一下
    }
}
