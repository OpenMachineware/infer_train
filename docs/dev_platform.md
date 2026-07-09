# InferTrain Hardware Platform Integration Guide


## 1. Overview

This guide explains how to add new hardware platform support (GPU/NPU/TPU) to the InferTrain engine.

Supported hardware types:
- Apple GPU (M-series chips)
- NVIDIA CUDA GPU
- AMD ROCm GPU
- NPU
- GPU
- DSP
- Other acceleration hardware


## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     InferTrain Engine                       │
├─────────────────────────────────────────────────────────────┤
│                     Operator Trait                          │
│              (Vendor only needs to implement this)          │
├─────────────────────────────────────────────────────────────┤
│                  Device Operator                            │
│               (CPU / GPU / NPU)                             │
├─────────────────────────────────────────────────────────────┤
│                  Tensor (Unified Data Format)               │
└─────────────────────────────────────────────────────────────┘
```


## 3. Core Interface

Vendors only need to implement the `Operator<T>` trait:

```rust
// src/ops/registry.rs

pub trait Operator<T: DType + Send + Sync>: Send + Sync {
    /// Operator name
    fn name(&self) -> &'static str;

    /// Forward propagation
    fn forward(&self, inputs: &[&Tensor<T>], attrs: &OpAttrs) -> Tensor<T>;

    /// Backward propagation (optional)
    fn backward(
        &self,
        _grad_output: &Tensor<T>,
        _inputs: &[&Tensor<T>],
        _attrs: &OpAttrs,
    ) -> Vec<Tensor<T>> {
        vec![]
    }

    /// Whether supports quantization
    fn supports_quantized(&self) -> bool {
        false
    }

    /// Device type
    fn device_type(&self) -> DeviceType {
        DeviceType::CPU
    }
}
```


## 4. Hardware Integration Approaches

### 4.1 Approach 1: C++ Operator Library (Most Common)

```
Vendor C++ Operator Library
    ↓
C API (extern "C")
    ↓
Rust FFI Wrapper
    ↓
Implement Operator Trait
    ↓
Register to Engine
```

### 4.2 Approach 2: Pure Rust Implementation

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
        // Call NPU SDK
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

### 4.3 Approach 3: CUDA/OpenCL

```rust
pub struct CUDAMatmul;

impl<T: DType + Send + Sync> Operator<T> for CUDAMatmul {
    fn name(&self) -> &'static str { "cuda_matmul" }
    
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // Call CUDA kernel
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


## 5. Platform-Specific Integration Steps

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

// Implement operator
pub struct MetalMatMul {
    device: MetalDevice,
    pipeline: ComputePipelineState,
}

impl<T: DType + Send + Sync> Operator<T> for MetalMatMul {
    fn name(&self) -> &'static str { "metal_matmul" }
    
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // 1. Create Metal buffers
        // 2. Encode compute commands
        // 3. Execute and wait
        // 4. Read results
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

// Use cust or cudarc crate
impl CudaDevice {
    pub fn new() -> Result<Self, String> {
        let device = Device::get_device(0)?;
        let context = Context::new(device)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(CudaDevice { context, stream })
    }
}
```

### 5.3 Huawei Ascend NPU

```rust
// src/device/ascend.rs

// Use CANN (Compute Architecture for Neural Networks)
// Need to link libascendcl.so

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
        // Initialize CANN
        unsafe {
            aclInit(...);
            aclrtCreateContext(&context, device_id);
        }
        Ok(AscendNPU { context, stream })
    }
}
```

### 5.4 Cambricon NPU

```rust
// src/device/cambricon.rs

// Use Cambricon CNToolkit
extern "C" {
    fn cnrtInit(...);
    fn cnrtMemcpy(...);
}

pub struct CambriconNPU {
    queue: *mut c_void,
    memory: *mut c_void,
}
```


## 6. Vendor Integration Steps

### Step 1: Create Device Module

```rust
// src/device/{vendor}.rs

pub struct VendorDevice {
    // Device handle
}

impl VendorDevice {
    pub fn new() -> Result<Self, String> {
        // Initialize device
    }
    
    pub fn allocate_memory(&self, size: usize) -> Result<*mut c_void, String> {
        // Allocate device memory
    }
    
    pub fn copy_to_device(&self, src: *const c_void, dst: *mut c_void, size: usize) {
        // Copy to device
    }
    
    pub fn copy_from_device(&self, src: *const c_void, dst: *mut c_void, size: usize) {
        // Copy from device
    }
}
```

### Step 2: Implement Operators

```rust
// src/device/{vendor}_ops.rs

pub struct VendorAdd {
    device: VendorDevice,
}

impl<T: DType + Send + Sync> Operator<T> for VendorAdd {
    fn name(&self) -> &'static str { "vendor_add" }
    
    fn forward(&self, inputs: &[&Tensor<T>], _attrs: &OpAttrs) -> Tensor<T> {
        // 1. Allocate device memory
        // 2. Copy input data
        // 3. Execute computation
        // 4. Copy back results
        // 5. Free device memory
        output
    }
}
```

### Step 3: Register to Engine

```rust
// src/lib.rs

pub fn register_vendor_ops(registry: &mut OperatorRegistry) {
    registry.register(VendorAdd::new());
    registry.register(VendorMatMul::new());
    registry.register(VendorConv2d::new());
    // ...
}
```

### Step 4: User Usage

```python
import infer_train_torch as it

# Enable NPU
it.enable_device("vendor_npu")

# Auto use NPU
model = MyModel()
x = torch.randn(1, 3, 224, 224)
y = model(x)  # ← Auto uses NPU
```


## 7. Device Selection Strategy

```rust
// src/device/selector.rs

pub struct DeviceSelector;

impl DeviceSelector {
    pub fn select_device() -> DeviceType {
        // Priority: NPU > GPU > CPU
        if Self::has_npu() {
            DeviceType::NPU
        } else if Self::has_gpu() {
            DeviceType::GPU
        } else {
            DeviceType::CPU
        }
    }
    
    pub fn has_npu() -> bool {
        // Detect if NPU exists
        false
    }
    
    pub fn has_gpu() -> bool {
        // Detect if GPU exists (Metal/CUDA/ROCm)
        false
    }
}
```


## 8. Vendor Integration Checklist

- [ ] Hardware SDK installed
- [ ] C API or Rust bindings available
- [ ] Core operators implemented (at least 10 common operators)
- [ ] Memory allocation/deallocation implemented
- [ ] Data transfer (Host ↔ Device) implemented
- [ ] Operators registered to engine
- [ ] Unit tests passed
- [ ] Performance benchmark testing


## 9. Core Operators to Implement

| Priority | Operator | Description |
|--------|------|------|
| P0 | add, sub, mul, div | Basic math |
| P0 | matmul | Matrix multiplication |
| P0 | conv2d | Convolution |
| P0 | relu, sigmoid, tanh | Activation functions |
| P0 | batch_norm | Batch normalization |
| P0 | max_pool, avg_pool | Pooling |
| P1 | linear | Linear layer |
| P1 | softmax | Softmax |
| P1 | reshape, transpose | Tensor manipulation |
| P1 | sum, mean | Reduction |
| P2 | conv_transpose | Transposed convolution |
| P2 | layer_norm | Layer normalization |
| P2 | embedding | Embedding |


## 10. Performance Recommendations

1. **Memory Reuse**: Use the engine's MemoryPool
2. **Asynchronous Execution**: Use streams/queues
3. **Batch Processing**: Merge small operations
4. **Mixed Precision**: Support FP16/BF16
5. **Operator Fusion**: Reduce memory access


## 11. FAQ

### Q: Must all operators be implemented?

A: No. The engine will fallback to CPU implementation. Recommended to implement P0 operators (10) first, then expand gradually.

### Q: How to debug hardware operators?

A: Use numerical gradient verification: `engine.grad_check(param_id, grad, eps)`

### Q: Must quantized versions be implemented?

A: Not mandatory. If hardware doesn't support quantization, only FP32/FP16 versions can be implemented.