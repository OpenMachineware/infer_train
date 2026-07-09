# InferTrain 硬件平台接入指南


## 1. 概述

本指南说明如何为 InferTrain 引擎添加新的硬件平台支持（GPU/NPU/TPU）。

支持的硬件类型：
- Apple GPU (M系列芯片)
- NVIDIA CUDA GPU
- AMD ROCm GPU
- NPU
- GPU
- DSP
- 其他加速硬件


## 2. 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                     InferTrain 引擎                         │
├─────────────────────────────────────────────────────────────┤
│                     Operator Trait                         │
│              (厂商只需实现这个接口)                          │
├─────────────────────────────────────────────────────────────┤
│                  Device Operator                           │
│               (CPU / GPU / NPU)                            │
├─────────────────────────────────────────────────────────────┤
│                  Tensor (统一数据格式)                      │
└─────────────────────────────────────────────────────────────┘
```


## 3. 核心接口

厂商只需要实现 `Operator<T>` trait：

```rust
// src/ops/registry.rs

pub trait Operator<T: DType + Send + Sync>: Send + Sync {
    /// 算子名称
    fn name(&self) -> &'static str;

    /// 前向传播
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T>;

    /// 反向传播（可选）
    fn backward(
        &self,
        _grad_output: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }

    /// 是否支持量化
    fn supports_quantized(&self) -> bool {
        false
    }

    /// 设备类型
    fn device_type(&self) -> DeviceType {
        DeviceType::CPU
    }
}
```


## 4. 硬件接入方式

### 4.1 方式一：C++ 算子库（最常用）

```
厂商 C++ 算子库
    ↓
C API (extern "C")
    ↓
Rust FFI 封装
    ↓
实现 Operator Trait
    ↓
注册到引擎
```

### 4.2 方式二：纯 Rust 实现

```rust
pub struct MyNPUAdd {
    handle: *mut c_void,
}

impl MyNPUAdd {
    pub fn new() -> Self {
        MyNPUAdd {
            handle: unsafe { npu_create() },
        }
    }
}

impl<T: DType + Send + Sync> Operator<T> for MyNPUAdd {
    fn name(&self) -> &'static str { "npu_add" }
    
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // 调用 NPU SDK
        unsafe {
            npu_add(
                self.handle,
                inputs[0].data_ptr(),
                inputs[1].data_ptr(),
                output.data_mut_ptr(),
                inputs[0].len(),
            );
        }
        output
    }
}
```

### 4.3 方式三：CUDA/OpenCL

```rust
pub struct CUDAMatmul;

impl<T: DType + Send + Sync> Operator<T> for CUDAMatmul {
    fn name(&self) -> &'static str { "cuda_matmul" }
    
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // 调用 CUDA kernel
        unsafe { 
            cuda_matmul_kernel(
                inputs[0].data_ptr(),
                inputs[1].data_ptr(),
                output.data_mut_ptr(),
                inputs[0].shape(),
                inputs[1].shape(),
            );
        }
        output
    }
}
```


## 5. 具体平台接入步骤

### 5.1 Apple GPU (Metal)

```rust
// src/device/metal.rs

use metal::*;

pub struct MetalDevice {
    device: Device,
    queue: CommandQueue,
}

impl MetalDevice {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let queue = device.new_command_queue();
        MetalDevice { device, queue }
    }
}

// 实现算子
pub struct MetalMatMul {
    device: MetalDevice,
    pipeline: ComputePipelineState,
}

impl<T: DType + Send + Sync> Operator<T> for MetalMatMul {
    fn name(&self) -> &'static str { "metal_matmul" }
    
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // 1. 创建 Metal 缓冲区
        // 2. 编码计算命令
        // 3. 执行并等待
        // 4. 读取结果
        output
    }
}
```

### 5.2 NVIDIA CUDA

```rust
// src/device/cuda.rs

use cust::prelude::*;

pub struct CudaDevice {
    context: Context,
    stream: Stream,
}

// 使用 cust 或 cudarc crate
impl CudaDevice {
    pub fn new() -> Result<Self, String> {
        let device = Device::get_device(0)?;
        let context = Context::new(device)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(CudaDevice { context, stream })
    }
}
```

### 5.3 华为昇腾 NPU

```rust
// src/device/ascend.rs

// 使用 CANN (Compute Architecture for Neural Networks)
// 需要链接 libascendcl.so

extern "C" {
    fn aclrtCreateContext(...);
    fn aclrtRun(...);
}

pub struct AscendNPU {
    context: *mut c_void,
    stream: *mut c_void,
}

impl AscendNPU {
    pub fn new() -> Result<Self, String> {
        // 初始化 CANN
        unsafe {
            aclInit(...);
            aclrtCreateContext(&context, device_id);
        }
        Ok(AscendNPU { context, stream })
    }
}
```

### 5.4 寒武纪 NPU

```rust
// src/device/cambricon.rs

// 使用寒武纪 CNToolkit
extern "C" {
    fn cnrtInit(...);
    fn cnrtMemcpy(...);
}

pub struct CambriconNPU {
    queue: *mut c_void,
    memory: *mut c_void,
}
```


## 6. 厂商接入步骤

### Step 1: 创建设备模块

```rust
// src/device/{vendor}.rs

pub struct VendorDevice {
    // 设备句柄
}

impl VendorDevice {
    pub fn new() -> Result<Self, String> {
        // 初始化设备
    }
    
    pub fn allocate_memory(&self, size: usize) -> Result<*mut c_void, String> {
        // 分配设备内存
    }
    
    pub fn copy_to_device(&self, src: *const c_void, dst: *mut c_void, size: usize) {
        // 拷贝到设备
    }
    
    pub fn copy_from_device(&self, src: *const c_void, dst: *mut c_void, size: usize) {
        // 从设备拷贝
    }
}
```

### Step 2: 实现算子

```rust
// src/device/{vendor}_ops.rs

pub struct VendorAdd {
    device: VendorDevice,
}

impl<T: DType + Send + Sync> Operator<T> for VendorAdd {
    fn name(&self) -> &'static str { "vendor_add" }
    
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // 1. 分配设备内存
        // 2. 拷贝输入数据
        // 3. 执行计算
        // 4. 拷贝回结果
        // 5. 释放设备内存
        output
    }
}
```

### Step 3: 注册到引擎

```rust
// src/lib.rs

pub fn register_vendor_ops(registry: &mut OperatorRegistry) {
    registry.register(VendorAdd::new());
    registry.register(VendorMatMul::new());
    registry.register(VendorConv2d::new());
    // ...
}
```

### Step 4: 用户使用

```python
import infer_train_torch as it

# 启用 NPU
it.enable_device("vendor_npu")

# 自动走 NPU
model = MyModel()
x = torch.randn(1, 3, 224, 224)
y = model(x)  # ← 自动走 NPU
```


## 7. 设备选择策略

```rust
// src/device/selector.rs

pub struct DeviceSelector;

impl DeviceSelector {
    pub fn select_device() -> DeviceType {
        // 优先级：NPU > GPU > CPU
        if Self::has_npu() {
            DeviceType::NPU
        } else if Self::has_gpu() {
            DeviceType::GPU
        } else {
            DeviceType::CPU
        }
    }
    
    pub fn has_npu() -> bool {
        // 检测是否有 NPU
        false
    }
    
    pub fn has_gpu() -> bool {
        // 检测是否有 GPU (Metal/CUDA/ROCm)
        false
    }
}
```


## 8. 厂商接入检查清单

- [ ] 硬件 SDK 安装完成
- [ ] C API 或 Rust 绑定可用
- [ ] 实现核心算子 (至少 10 个常用算子)
- [ ] 内存分配/释放实现
- [ ] 数据传输 (Host ↔ Device) 实现
- [ ] 算子注册到引擎
- [ ] 单元测试通过
- [ ] 性能基准测试


## 9. 需要实现的核心算子

| 优先级 | 算子 | 说明 |
|--------|------|------|
| P0 | add, sub, mul, div | 基础数学 |
| P0 | matmul | 矩阵乘法 |
| P0 | conv2d | 卷积 |
| P0 | relu, sigmoid, tanh | 激活函数 |
| P0 | batch_norm | 批归一化 |
| P0 | max_pool, avg_pool | 池化 |
| P1 | matmul | 线性层 |
| P1 | softmax | Softmax |
| P1 | reshape, transpose | 张量操作 |
| P1 | sum, mean | 归约 |
| P2 | conv_transpose | 转置卷积 |
| P2 | layer_norm | 层归一化 |
| P2 | embedding | 词嵌入 |


## 10. 性能建议

1. **内存复用**：使用引擎的 MemoryPool
2. **异步执行**：使用流/队列
3. **批量处理**：尽量合并小操作
4. **混合精度**：支持 FP16/BF16
5. **算子融合**：减少内存访问


## 11. 常见问题

### Q: 必须实现所有算子吗？

A: 不需要。引擎会 fallback 到 CPU 实现。建议先实现 P0 算子（10 个），再逐步扩展。

### Q: 如何调试硬件算子？

A: 可以用数值梯度验证：`engine.grad_check(param_id, grad, eps)`

### Q: 量化版本必须实现吗？

A: 不强制。如果硬件不支持量化，可以只实现 FP32/FP16 版本。
