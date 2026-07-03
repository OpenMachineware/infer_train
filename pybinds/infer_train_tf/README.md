# InferTrain Engine for TensorFlow

## 概述

InferTrain 是一个用 C++ 和 Rust 编写的推训一体引擎，通过 FFI 暴露给 Python。
本插件为 TensorFlow 版本，提供 PyTensor 类用于高性能张量运算。

<div style="font-size: 20px; font-weight: bold; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background-color: #f9f9f9;">

⚠️ 重要提示<br><br>

本插件不是无感侵入式的！<br><br>

TensorFlow 的 Python 算子（如 + - * / 和 tf.math.add）无法被覆盖。<br>
用户必须：<br>
&nbsp;&nbsp;1. 用 <code>tf_to_pytensor()</code> 将 tf.Tensor 转成 PyTensor<br>
&nbsp;&nbsp;2. 调用 PyTensor 的算子方法（如 a.add(b)）<br>
&nbsp;&nbsp;3. 用 <code>pytensor_to_tf()</code> 转回 tf.Tensor<br><br>

不要问为什么 a + b 不走引擎，文档已经写清楚了。<br><br>

如果你想要无感侵入，请用 <code>infer_train_torch</code>。

</div>

注意：TensorFlow 的 Python 算子无法被完全覆盖（因为 TF 核心是 C++ 实现），
因此本插件提供 PyTensor 类 + tf_to_pytensor / pytensor_to_tf 转换函数，
用户需要显式转换张量。

## 架构原理

<div style="font-family: sans-serif; font-size: 18px; font-weight: bold; width: 600px; border: 2px solid #333; border-radius: 8px; overflow: hidden;">
  
  <div style="padding: 15px; background-color: #e3f2fd; border-bottom: 2px solid #333; text-align: center;">
    Python 层 (TensorFlow)
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

## 安装

`pip install infer_train_tf`


## 使用方式

### 方式 1：从 tf.Tensor 转换（推荐）

```
import tensorflow as tf
import infer_train_tf as it

# ⚠️ 警告：不要用 a + b，用 a.add(b)
# 具体原因见 README

# 1. 创建 tf.Tensor
tf_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tf_b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 2. 转成 PyTensor
a = it.tf_to_pytensor(tf_a)
b = it.tf_to_pytensor(tf_b)

# 3. 使用 InferTrain 算子（必须显式调用）
c = a.add(b)          # ✅ 正确
# c = a + b           # ❌ 错误！走的是 TF 原生，不是 InferTrain

d = a.matmul(b)       # ✅ 正确
e = a.relu()          # ✅ 正确

# 4. 转回 tf.Tensor
result = it.pytensor_to_tf(c)
print(result.numpy())
```

### 方式 2：直接用 PyTensor 创建

```
import infer_train_tf as it

a = it.PyTensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = it.PyTensor([5.0, 6.0, 7.0, 8.0], [2, 2])
c = a.add(b)
print(c.data())  # [6.0, 8.0, 10.0, 12.0]
```

## API 参考

```
tf_to_pytensor(t: tf.Tensor) -> PyTensor

将 tf.Tensor 转成 PyTensor（只支持 CPU）。

a = it.tf_to_pytensor(tf.constant([[1.0, 2.0]]))

pytensor_to_tf(t: PyTensor) -> tf.Tensor

将 PyTensor 转回 tf.Tensor。

result = it.pytensor_to_tf(a)
```


## PyTensor 类

### 构造函数

```
PyTensor(data: List[float], shape: List[int], dtype: str = "f32", scale: Optional[float] = None, zero_point: Optional[float] = None)
```

- data: 一维数据列表
- shape: 张量形状
- dtype: f32, f64, f16, bf16, i8
- scale: 量化 scale（仅 i8 需要）
- zero_point: 量化 zero_point（仅 i8 需要）

### 属性

- t.data()        # 数据列表
- t.shape()       # 形状
- t.dtype()       # 数据类型
- t.scale()       # 量化 scale
- t.zero_point()  # 量化 zero_point

## 支持的算子

### 数学：
```
add, sub, mul, div, pow, exp, sqrt, log, log2, log10
abs, neg, clamp, floor, ceil, round
```

### 规约：
```
sum, mean, max_all, min_all, prod_all, var, std
```

### 矩阵：
```
matmul, batch_matmul, vec_matmul, transpose
```

### NN：
```
conv1d, conv2d, conv3d
maxpool1d, maxpool2d, maxpool3d
avgpool1d, avgpool2d, avgpool3d
batchnorm1d, batchnorm2d
layernorm, rmsnorm
instancenorm2d, groupnorm
linear, embedding, dropout
```

### 激活：
```
relu, leaky_relu, elu, gelu, relu6
sigmoid, tanh, silu, hard_swish, hard_sigmoid
softplus, softshrink, celu
softmax, log_softmax
```

### 张量操作：
```
slice, cat, cumsum, cumprod
```

### 注意力：
```
scaled_dot_product_attention, multi_head_attention, rotary_embedding
```

### 量化：
```
quantized_add, quantized_sub, quantized_mul, quantized_div,
quantized_exp, quantized_sqrt, quantized_abs, quantized_neg, quantized_clamp,
quantized_conv1d, quantized_conv2d, quantized_conv3d,
quantized_maxpool1d, quantized_maxpool2d, quantized_maxpool3d,
quantized_avgpool1d, quantized_avgpool2d, quantized_avgpool3d,
quantized_batchnorm1d, quantized_batchnorm2d,
quantized_layernorm, quantized_rmsnorm,
quantized_linear, quantized_embedding,
quantized_relu, quantized_leaky_relu, quantized_elu, quantized_gelu, quantized_relu6,
quantized_sigmoid, quantized_tanh, quantized_silu,
quantized_hard_swish, quantized_hard_sigmoid,
quantized_softmax, quantized_log_softmax
```
### 损失：
```
cross_entropy_loss, mse_loss, l1_loss, bce_loss
```

## 与 PyTorch 版本的区别

| 框架 | 覆盖原生算子 | 使用方式 |
| :--- | :--- | :--- |
| PyTorch | 自动覆盖 | import 后原生算子自动走引擎 |
| JAX | 不能覆盖 | 需要显式转换张量 |
| TensorFlow | 不能覆盖 | 需要显式转换张量 |

__原因：TensorFlow 核心是 C++ 实现，Python 层只是薄封装，无法有效覆盖。__


## 环境要求

- Python >= 3.8
- TensorFlow >= 2.13.0
- CPU only（GPU 暂不支持）

## 许可证

Apache-2.0


## 贡献

欢迎提交 Issue 和 __Pull Request__。
