# InferTrain Engine for PyTorch

## 概述

InferTrain 是一个用 C++ 和 Rust 编写的推训一体引擎，通过 FFI 暴露给 Python，支持 PyTorch 框架的无感接入。

## 架构原理

<div style="font-family: sans-serif; font-size: 18px; font-weight: bold; width: 600px; border: 2px solid #333; border-radius: 8px; overflow: hidden;">
  
  <div style="padding: 15px; background-color: #e3f2fd; border-bottom: 2px solid #333; text-align: center;">
    Python 层 (PyTorch)
  </div>
  
  <div style="padding: 15px; background-color: #fff3e0; border-bottom: 2px solid #333; text-align: center;">
    PyO3 绑定 (Rust)
  </div>
  
  <div style="padding: 15px; background-color: #e8f5e9; border-bottom: 2px solid #333; text-align: center;">
    C API 接口
  </div>
  
  <div style="padding: 15px; background-color: #f3e5f5; text-align: center; line-height: 1.6;">
    C++ 算子库<br>
    <span style="font-size: 16px; font-weight: normal;">
      - 100+ 算子 (CPU / 量化)<br>
      - 精度支持：FP32 / FP64 / FP16 / BF16 / I8
    </span>
  </div>

</div>

### 工作流程

1. Python 层创建 PyTensor 对象，传入数据和 shape
2. PyO3 将 Python 对象转换为 Rust 的 Tensor 结构
3. Rust FFI 调用 C API 接口
4. C++ 算子执行计算
5. 结果逐层返回

### 算子支持

- 数学运算: add, sub, mul, div, pow, exp, sqrt, log, abs, neg, clamp, floor, ceil, round
- 规约运算: sum, mean, max, min, std, var, prod
- 矩阵运算: matmul, batch_matmul, vec_matmul, transpose
- 神经网络: conv1d/2d/3d, maxpool1d/2d/3d, avgpool1d/2d/3d, batchnorm1d/2d, layernorm, rmsnorm, instancenorm2d, groupnorm, linear, embedding, dropout
- 激活函数: relu, leaky_relu, elu, gelu, relu6, sigmoid, tanh, silu, hard_swish, hard_sigmoid, softplus, softshrink, celu, softmax, log_softmax
- 张量操作: transpose, slice, cat, cumsum, cumprod
- 注意力: scaled_dot_product_attention, multi_head_attention, rotary_embedding
- 损失函数: cross_entropy_loss, mse_loss, l1_loss, bce_loss
- 量化算子: 所有算子的 I8 量化版本 (quantized_*)

## 安装

```bash
pip install infer_train_torch          # PyTorch 版本
```

## 快速开始

### PyTorch
```python
import torch
import infer_train_torch as it

# 创建张量
a = it.PyTensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = it.PyTensor([5.0, 6.0, 7.0, 8.0], [2, 2])

# 加法
c = a.add(b)
print(c.data())  # [6.0, 8.0, 10.0, 12.0]

# 矩阵乘法
d = a.matmul(b)
print(d.data())  # [19.0, 22.0, 43.0, 50.0]

# ReLU
e = it.PyTensor([-1.0, 2.0, -3.0, 4.0], [2, 2])
f = e.relu()
print(f.data())  # [0.0, 2.0, 0.0, 4.0]

# 量化张量
g = it.PyTensor([1, 2, 3, 4], [2, 2], dtype="i8", scale=0.01, zero_point=0.0)
print(g.dtype())  # i8
print(g.scale())  # 0.01

# 卷积
input = it.PyTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3])
weight = it.PyTensor([1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2])
output = input.conv2d(weight, None, 1, 0, 1, 1)
print(output.shape())  # [1, 1, 2, 2]

# 张量拼接
a = it.PyTensor([1.0, 2.0], [1, 2])
b = it.PyTensor([3.0, 4.0], [1, 2])
c = it.PyTensor.cat([a, b], 0)
print(c.data())  # [1.0, 2.0, 3.0, 4.0]

# 量化运算
a = it.PyTensor([1, 2, 3, 4], [2, 2], dtype="i8", scale=0.01, zero_point=0.0)
b = it.PyTensor([5, 6, 7, 8], [2, 2], dtype="i8", scale=0.01, zero_point=0.0)
c = a.quantized_add(b)
print(c.dtype())  # i8
```

## API 参考

### PyTensor 类

#### 构造函数
```python
PyTensor(data: List[float], shape: List[int], dtype: str = "f32", scale: Optional[float] = None, zero_point: Optional[float] = None)
```

- data: 张量数据（一维列表）
- shape: 张量形状
- dtype: 数据类型 (f32, f64, f16, bf16, i8)
- scale: 量化 scale（仅 i8 时需要）
- zero_point: 量化 zero_point（仅 i8 时需要）

#### 属性

```python
t.data() -> List[float]
t.shape() -> List[int]
t.dtype() -> str
t.scale() -> Optional[float]
t.zero_point() -> Optional[float]
```

#### 数学算子

```python
t.add(other: PyTensor) -> PyTensor
t.sub(other: PyTensor) -> PyTensor
t.mul(other: PyTensor) -> PyTensor
t.div(other: PyTensor) -> PyTensor
t.pow(other: PyTensor) -> PyTensor
t.exp() -> PyTensor
t.sqrt() -> PyTensor
t.log() -> PyTensor
t.log2() -> PyTensor
t.log10() -> PyTensor
t.abs() -> PyTensor
t.neg() -> PyTensor
t.clamp(min_val: float, max_val: float) -> PyTensor
t.floor() -> PyTensor
t.ceil() -> PyTensor
t.round() -> PyTensor
```

#### 规约算子

```python
t.sum(dims: List[int] = [], keepdim: bool = False) -> PyTensor
t.mean(dims: List[int] = [], keepdim: bool = False) -> PyTensor
t.max_all() -> PyTensor
t.min_all() -> PyTensor
t.prod_all() -> PyTensor
t.var(unbiased: bool = False) -> PyTensor
t.std(unbiased: bool = False) -> PyTensor
```

#### 矩阵算子

```python
t.matmul(other: PyTensor) -> PyTensor
t.batch_matmul(other: PyTensor) -> PyTensor
t.vec_matmul(mat: PyTensor) -> PyTensor
t.transpose() -> PyTensor
```

#### NN算子

```python
t.conv1d(weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> PyTensor
t.conv2d(weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> PyTensor
t.conv3d(weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> PyTensor
t.maxpool1d(kernel_size, stride=-1, padding=0) -> PyTensor
t.maxpool2d(kernel_size, stride=-1, padding=0) -> PyTensor
t.maxpool3d(kernel_size, stride=-1, padding=0) -> PyTensor
t.avgpool1d(kernel_size, stride=-1, padding=0) -> PyTensor
t.avgpool2d(kernel_size, stride=-1, padding=0) -> PyTensor
t.avgpool3d(kernel_size, stride=-1, padding=0) -> PyTensor
t.batchnorm1d(weight, bias, running_mean, running_var, eps=1e-5) -> PyTensor
t.batchnorm2d(weight, bias, running_mean, running_var, eps=1e-5) -> PyTensor
t.layernorm(weight, bias, eps=1e-5) -> PyTensor
t.rmsnorm(weight, eps=1e-6) -> PyTensor
t.instancenorm2d(weight, bias, eps=1e-5) -> PyTensor
t.groupnorm(weight, bias, num_groups, eps=1e-5) -> PyTensor
t.linear(weight, bias=None) -> PyTensor
t.embedding(indices, padding_idx=-1) -> PyTensor
t.dropout(p=0.5) -> PyTensor
```

#### 激活函数

```python
t.relu() -> PyTensor
t.leaky_relu(alpha=0.01) -> PyTensor
t.elu(alpha=1.0) -> PyTensor
t.gelu() -> PyTensor
t.relu6() -> PyTensor
t.sigmoid() -> PyTensor
t.tanh() -> PyTensor
t.silu() -> PyTensor
t.hard_swish() -> PyTensor
t.hard_sigmoid() -> PyTensor
t.softplus(beta=1.0, threshold=20.0) -> PyTensor
t.softshrink(lambda=0.5) -> PyTensor
t.celu(alpha=1.0) -> PyTensor
t.softmax(dim=-1) -> PyTensor
t.log_softmax(dim=-1) -> PyTensor
```

#### 张量操作

```python
t.slice(dim, start, end, step=1) -> PyTensor
PyTensor.cat(tensors, dim) -> PyTensor  (静态方法)
t.cumsum(dim=-1) -> PyTensor
t.cumprod(dim=-1) -> PyTensor
```

#### 量化算子

```python
t.quantized_add(other) -> PyTensor
t.quantized_sub(other) -> PyTensor
t.quantized_mul(other) -> PyTensor
t.quantized_div(other) -> PyTensor
t.quantized_exp() -> PyTensor
t.quantized_sqrt() -> PyTensor
t.quantized_abs() -> PyTensor
t.quantized_neg() -> PyTensor
t.quantized_clamp(min_val, max_val) -> PyTensor
t.quantized_conv1d(weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> PyTensor
t.quantized_conv2d(weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> PyTensor
t.quantized_conv3d(weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> PyTensor
t.quantized_maxpool1d(kernel_size, stride=-1, padding=0) -> PyTensor
t.quantized_maxpool2d(kernel_size, stride=-1, padding=0) -> PyTensor
t.quantized_maxpool3d(kernel_size, stride=-1, padding=0) -> PyTensor
t.quantized_avgpool1d(kernel_size, stride=-1, padding=0) -> PyTensor
t.quantized_avgpool2d(kernel_size, stride=-1, padding=0) -> PyTensor
t.quantized_avgpool3d(kernel_size, stride=-1, padding=0) -> PyTensor
t.quantized_batchnorm1d(weight, bias, running_mean, running_var, eps=1e-5) -> PyTensor
t.quantized_batchnorm2d(weight, bias, running_mean, running_var, eps=1e-5) -> PyTensor
t.quantized_layernorm(weight, bias, eps=1e-5) -> PyTensor
t.quantized_rmsnorm(weight, eps=1e-6) -> PyTensor
t.quantized_linear(weight, bias=None) -> PyTensor
t.quantized_embedding(indices, padding_idx=-1) -> PyTensor
t.quantized_relu() -> PyTensor
t.quantized_leaky_relu(alpha=0.01) -> PyTensor
t.quantized_elu(alpha=1.0) -> PyTensor
t.quantized_gelu() -> PyTensor
t.quantized_relu6() -> PyTensor
t.quantized_sigmoid() -> PyTensor
t.quantized_tanh() -> PyTensor
t.quantized_silu() -> PyTensor
t.quantized_hard_swish() -> PyTensor
t.quantized_hard_sigmoid() -> PyTensor
t.quantized_softmax(dim=-1) -> PyTensor
t.quantized_log_softmax(dim=-1) -> PyTensor
```

#### 注意力算子

```python
t.scaled_dot_product_attention(key, value, mask=None, scale=-1.0, is_causal=False, dropout_p=0.0) -> PyTensor
t.multi_head_attention(key, value, mask=None, num_heads=8, scale=-1.0, is_causal=False, dropout_p=0.0) -> PyTensor
t.rotary_embedding(cos, sin) -> PyTensor
t.quantized_scaled_dot_product_attention(key, value, mask=None, scale=-1.0, is_causal=False, dropout_p=0.0) -> PyTensor
t.quantized_multi_head_attention(key, value, mask=None, num_heads=8, scale=-1.0, is_causal=False, dropout_p=0.0) -> PyTensor
t.quantized_rotary_embedding(cos, sin) -> PyTensor
```

#### 损失函数

```python
t.cross_entropy_loss(target, reduction=True) -> PyTensor
t.mse_loss(target, reduction=True) -> PyTensor
t.l1_loss(target, reduction=True) -> PyTensor
t.bce_loss(target, reduction=True, eps=1e-7) -> PyTensor
```

## 构建开发

### 构建 C++ 算子库

```bash
cd operators
meson setup build
meson compile -C build
```

### 构建 Python 包

```bash
cd pybinds/infer_train_torch
maturin build --release
pip install target/wheels/*.whl
```

### 环境要求

* Python >= 3.8
* PyTorch >= 2.0.0 (PyTorch 插件)
* CPU only (GPU 支持将 fallback 到原生框架)

### 许可证

Apache-2.0

### 贡献

欢迎提交 Issue 和 __Pull Request__。
