# InferTrain 算子开发文档

## 第一部分：如何添加一个新算子

假设你要添加一个算子 `mylayer`，输入一个张量，输出一个张量，支持 FP32/FP64/FP16/BF16/I8。

---

### 1. C++ 层

#### 1.1 头文件

创建 `operators/include/infer_train/math/mylayer.hpp`：

```c++
#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"

namespace infer_train {

template<typename T>
Tensor<T> mylayer(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> output(input.shape);

    for (size_t i = 0; i < output.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float y = x * x + 1.0f;  // 你的计算逻辑
        output.data[i] = Conv::from_float(y);
    }

    return output;
}

} // namespace infer_train
```

#### 1.2 量化版本（可选）

创建 `operators/include/infer_train/math/mylayer_q.hpp`：

```c++
#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/mylayer.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_mylayer(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value, "quantized_mylayer only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_output = mylayer<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(fp32_output, scale, zero_point);
    return quantize<T>(fp32_output, scale, zero_point);
}

} // namespace infer_train
```

#### 1.3 实现文件

创建 `operators/src/math/mylayer.cpp`：

```c++
#include "infer_train/math/mylayer.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> mylayer(const Tensor<F32>&);
template Tensor<F64> mylayer(const Tensor<F64>&);
template Tensor<F16> mylayer(const Tensor<F16>&);
template Tensor<BF16> mylayer(const Tensor<BF16>&);

} // namespace infer_train
```

#### 1.4 量化实现文件

创建 `operators/src/math/mylayer_q.cpp`：

```c++
#include "infer_train/math/mylayer_q.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<I8> quantized_mylayer(const Tensor<I8>&);

} // namespace infer_train
```

#### 1.5 注册到聚合头文件

修改 `operators/include/infer_train/math.hpp`：

```c++
#pragma once
#include "infer_train/math/add.hpp"
#include "infer_train/math/mylayer.hpp"   // 新增
```

修改 `operators/include/infer_train/math_q.hpp`：

```c++
#pragma once
#include "infer_train/math/add_q.hpp"
#include "infer_train/math/mylayer_q.hpp"   // 新增
```

#### 1.6 更新 meson.build

在 `operators/meson.build` 的 `sources` 中添加：

```text
'src/math/mylayer.cpp',
'src/math/mylayer_q.cpp',
```

---

### 2. C API 层

#### 2.1 头文件声明

修改 `operators/include/infer_train/c_interface.h`：

```c++
// 在合适的位置添加
it_tensor_t* it_mylayer(const it_tensor_t* input);
```

#### 2.2 C API 实现（浮点）

创建或修改 `src/c_api/c_api_math.cpp`：

```c++
extern "C" it_tensor* it_mylayer(const it_tensor* input) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = mylayer(t);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = mylayer(t);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = mylayer(t);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = mylayer(t);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        case IT_DTYPE_I8: {
            auto t = to_cpp_tensor_with_params<I8>(input);
            auto result = quantized_mylayer(t);
            return from_cpp_tensor(result, IT_DTYPE_I8);
        }
        default: return nullptr;
    }
}
```

---

### 3. Rust FFI 层

#### 3.1 FFI 声明

修改 `src/ffi.rs`，在 `extern "C"` 块中添加：

```rust
pub fn it_mylayer(input: *const it_tensor) -> *mut it_tensor;
```

#### 3.2 Rust 封装

在 `impl Tensor` 中添加：

```rust
pub fn mylayer(&self) -> Tensor {
    let ptr = unsafe { it_mylayer(self.ptr) };
    Tensor { ptr }
}
```

---

### 4. PyO3 层

#### 4.1 Python 方法

修改 `src/pytensor.rs`，在 `#[pymethods]` 块中添加：

```rust
fn mylayer(&self) -> PyTensor {
    PyTensor {
        inner: self.inner.mylayer(),
    }
}
```

#### 4.2 可选：添加参数（如果有参数）

```rust
#[pyo3(signature = (param1, param2=1.0))]
fn mylayer_with_params(&self, param1: f32, param2: f32) -> PyTensor {
    PyTensor {
        inner: self.inner.mylayer_with_params(param1, param2),
    }
}
```

---

### 5. 三个插件的导出

PyTorch/TensorFlow/JAX 插件共享同一个 Rust 库，所以 PyO3 层添加后，三个插件自动支持。

不需要额外修改每个插件的 `__init__.py`。

用户使用：

- PyTorch
```python
import infer_train_torch as it
a = it.PyTensor([1.0, 2.0], [1, 2])
b = a.mylayer()
```

- TensorFlow（相同 API）
```python
import infer_train_tf as it
a = it.PyTensor([1.0, 2.0], [1, 2])
b = a.mylayer()
```

- JAX（相同 API）
```python
import infer_train_jax as it
a = it.PyTensor([1.0, 2.0], [1, 2])
b = a.mylayer()
```
---

### 6. PyTorch 无感侵入的 Python 层

#### 修改 `python/infer_train_torch/__init__.py`

在 `__init__.py` 中添加算子覆盖逻辑：

```python
# 在文件末尾，添加对应的覆盖函数
_original_mylayer = torch.mylayer  # 或对应的 torch 函数

def _it_mylayer(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.mylayer()
            return pytensor_to_torch(out)
        except Exception:
            return _original_mylayer(input)
    return _original_mylayer(input)

# 覆盖 torch 函数
torch.mylayer = _it_mylayer
```

```python
# 如果是 Tensor 的方法（如 .add 是 Tensor.__add__）
# 覆盖方式：
_original_add = Tensor.__add__
Tensor.__add__ = _it_add

# 如果是 torch.nn.functional 下的函数：
_original_relu = F.relu
F.relu = _it_relu
```
---

### 7. PyTorch 无感侵入的覆盖模式

对于 PyTorch，有三种覆盖方式：

| 算子类型 | 覆盖方式 | 示例 |
|---------|---------|------|
| Tensor 方法 | 覆盖 `Tensor.__xxx__` | `Tensor.__add__` |
| torch 模块函数 | 覆盖 `torch.xxx` | `torch.matmul` |
| torch.nn.functional | 覆盖 `F.xxx` | `F.relu`, `F.conv2d` |

覆盖函数模板：

```python
def _it_xxx(self, *args, **kwargs):
    # 只在 CPU 上使用引擎
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            # 调用 PyTensor 方法
            result = t.xxx(*args, **kwargs)
            return pytensor_to_torch(result)
        except Exception:
            # fallback 到 PyTorch 原生
            return _original_xxx(self, *args, **kwargs)
    return _original_xxx(self, *args, **kwargs)
```
---

### 8. 文件修改清单

新增文件：
- operators/include/infer_train/math/mylayer.hpp
- operators/include/infer_train/math/mylayer_q.hpp
- operators/src/math/mylayer.cpp
- operators/src/math/mylayer_q.cpp

修改文件：
- operators/include/infer_train/math.hpp
- operators/include/infer_train/math_q.hpp
- operators/meson.build
- operators/include/infer_train/c_interface.h
- src/c_api/c_api_math.cpp (或对应模块)
- src/ffi.rs
- src/pytensor.rs



## 第二部分：如何添加硬件支持

假设要添加一个 NPU 支持。

---

### 1. 整体架构

<div style="display:flex; flex-direction:column; gap:8px; font-family:sans-serif; color:#333;">
  <div style="background:#e3f2fd; border-left:4px solid #1976d2; padding:12px 16px; border-radius:4px; font-weight:bold;">Python 层</div>
  <div style="background:#f3e5f5; border-left:4px solid #7b1fa2; padding:12px 16px; border-radius:4px; font-weight:bold;">Rust FFI</div>
  <div style="background:#fff3e0; border-left:4px solid #f57c00; padding:12px 16px; border-radius:4px; font-weight:bold;">C API</div>
  <div style="display:flex; gap:8px;">
    <div style="flex:1; background:#fff3e0; border-left:4px solid #f57c00; padding:12px 16px; border-radius:4px; font-weight:bold; text-align:center;">C++ CPU 算子</div>
    <div style="flex:1; background:#fff3e0; border-left:4px solid #f57c00; padding:12px 16px; border-radius:4px; font-weight:bold; text-align:center;">NPU 算子</div>
  </div>
  <div style="background:#e8f5e9; border-left:4px solid #388e3c; padding:12px 16px; border-radius:4px; font-weight:bold;">NPU Driver / SDK</div>
</div>


---

### 2. 需要做的事情

#### 2.1 抽象算子接口

每个算子需要 CPU 和 NPU 两个实现：

```c++
namespace infer_train {

template<typename T>
Tensor<T> matmul_cpu(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> matmul_npu(const Tensor<T>& a, const Tensor<T>& b);

// 统一入口，根据设备选择实现
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.device == Device::NPU) {
        return matmul_npu(a, b);
    } else {
        return matmul_cpu(a, b);
    }
}

}
```

#### 2.2 张量增加设备信息

```c++
struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;
    Device device = Device::CPU;  // 新增
    void* npu_handle = nullptr;   // NPU 内存句柄
};
```

#### 2.3 内存管理

CPU 和 NPU 内存分开管理：

- CPU：`std::vector` 自动管理
- NPU：通过 `NPU SDK` 分配/释放，需要手动管理

```c++
struct NPUContext {
    void* alloc(size_t bytes);
    void free(void* ptr);
    void memcpy_host_to_device(void* dst, const void* src, size_t bytes);
    void memcpy_device_to_host(void* dst, const void* src, size_t bytes);
};
```

#### 2.4 算子实现

NPU 算子通常调用厂商 SDK：

```c++
template<typename T>
Tensor<T> matmul_npu(const Tensor<T>& a, const Tensor<T>& b) {
    // 1. 获取 NPU 句柄
    // 2. 调用 NPU SDK 的 matmul 函数
    // 3. 返回结果
}
```

#### 2.5 设备选择策略

简单策略：用户指定
```c++
a.to_npu()
a.to_cpu()
```

自动策略：根据输入设备自动选择

#### 2.6 异构执行

混合执行：部分算子在 CPU，部分在 NPU

---

### 3. 实现难度

| 层级 | 工作量 | 说明 |
|------|--------|------|
| 抽象接口 | 小 | 每个算子加一个 if 判断 |
| 张量设备字段 | 小 | 加两个字段 |
| 内存管理 | 中 | 需要 RAII 管理 NPU 内存 |
| 算子实现 | 大 | 每个算子需要 NPU 版本 |
| 厂商 SDK 集成 | 大 | 依赖厂商库 |

---

### 4. 优先级建议

1. 先确定支持的 NPU 型号和 SDK
2. 实现内存管理
3. 实现最常用的算子（matmul, add, conv2d, relu）
4. 逐步扩展

---

### 5. 用户使用方式

```python
a = PyTensor([1.0, 2.0], [1, 2])
a.to_npu()
b = a.matmul(c)   # 自动走 NPU
```

或者自动选择：

```python
a = PyTensor([1.0, 2.0], [1, 2])
b = PyTensor([3.0, 4.0], [2, 1])
c = a.matmul(b)   # 如果 a 和 b 在 NPU，走 NPU；否则走 CPU
```

__值得注意的是， PyTorch 支持无感侵入，可以让用户只增加一行import，代码无需修改，
你只需要完善算子库就行__

```python
import torch
import infer_train_torch as it

# 自动走 InferTrain 引擎（无感侵入）
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 这些操作自动走引擎
c = a + b          # 对应 Tensor.__add__
d = a @ b          # 对应 torch.matmul
e = a.relu()       # 对应 torch.relu
f = F.conv2d(x, w) # 对应 F.conv2d
```


TF 和 JAX 则需要显式转换：
```python
# TensorFlow（显式转换）
import tensorflow as tf
import infer_train_tf as it

tf_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
a = it.tf_to_pytensor(tf_a)
b = a.add(c)        # 必须显式调用方法

# JAX（显式转换）
import jax.numpy as jnp
import infer_train_jax as it

jax_a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
a = it.jax_to_pytensor(jax_a)
b = a.add(c)        # 必须显式调用方法
```
