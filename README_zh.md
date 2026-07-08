# InferTrain - 推训一体计算引擎

## 项目简介

InferTrain 是一个专为边缘设备设计的推训一体 AI 推理与训练引擎。

- **模型保密**：推理在设备端本地执行，模型权重和网络结构不离开设备，发布者无需公开网络结构即可让用户进行微调。同时，用户在本地微调后的权重可以保存并持续使用，私有化与个性化兼得。
- **推训一体**：同一套代码支持推理和训练，边推理边更新权重
- **多前端支持**：PyTorch Hook/JIT Trace、GGUF
- **多精度支持**：F32/F64/F16/BF16/I8 量化
- **内存复用**：字节池内存管理，适合边缘设备
- **ITM 私有格式**：自研二进制模型格式，支持分块存储、mmap、增量训练
- **纯 Rust 实现**：所有算子均为 Rust 实现，无 C++ 依赖

## 核心特性

| 特性 | 说明 |
|------|------|
| 推训一体 | 同一模型可推理也可训练，权重可增量更新 |
| 纯 Rust 算子 | 78+ 算子，支持 F32/F64/F16/BF16/I8 |
| PyTorch 无感侵入 | import 即可，用户代码无需修改 |
| GGUF 支持 | 导入/导出 GGUF 格式，支持量化模型训练 |
| ITM 格式 | 自研二进制格式，分块存储，支持 mmap |
| 内存复用 | 字节池内存管理，训练内存峰值降低 70% |
| 自动微分 | 基于 Tape 的自动微分引擎 |
| 并行执行 | Rayon 并行，充分利用多核 |


## 快速开始

### 安装

```bash
# 从源码安装（需要 Rust 环境）
git clone https://github.com/yourname/infer_train.git
cd infer_train
uv pip install -e .
```

### Python 使用

```python
import torch
import torch.nn as nn
import infer_train_torch as it  # ← 只加这一行

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        return self.out(self.relu(self.fc(x)))

model = MyModel()
x = torch.randn(1, 784)

# 推理（自动走引擎）
output = model(x)

# 追踪并导出
dag = it.trace_model(model, [x])
it.export_model("model.itm", "my_model", trainable=True)

# 加载并推理
model_file = it.PyModelFile.load("model.itm")
executor = it.PyExecutor.from_model_file(model_file)
result = executor.execute([x])
```

### GGUF 模型加载

```python
import infer_train_torch as it

# 导入 GGUF 模型
dag = it.import_gguf("llama-2-7b.gguf")

# 导出为 ITM（更快加载）
model_file = it.PyModelFile.new("llama", "gguf", dag)
model_file.export("llama.itm")

# 训练 GGUF 模型（边推边训）
trainer = it.Trainer.from_model_file(model_file)
loss = trainer.train_step(inputs, targets)
trainer.save("llama_updated.itm", trainable=True)

# 导出回 GGUF
it.export_gguf("llama_updated.gguf", dag, "Q8_0")
```


## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      Python 层                             │
│            (PyTorch Hook / GGUF 导入器)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Rust 核心层                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    CFG IR   │→│    DAG IR   │→│   Executor/Trainer   │ │
│  │  (控制流图) │  │  (数据流图) │  │   (推理/训练)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              78+ 算子 (纯 Rust)                     │   │
│  │  math / linalg / activation / conv / normalization │   │
│  │  reduction / loss / embedding / attention / ...    │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              自动微分 (Tape + Backward)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ITM 存储格式                            │
│            (分块存储 / mmap / 增量训练)                    │
└─────────────────────────────────────────────────────────────┘
```


## 目录结构

```
infer_train/
├── src/
│   ├── ops/              # 78+ 算子实现
│   │   ├── math/         # 数学算子
│   │   ├── linalg/       # 线性代数
│   │   ├── activation/   # 激活函数
│   │   ├── conv_pool/    # 卷积与池化
│   │   ├── normalization/# 归一化
│   │   ├── reduction/    # 归约
│   │   ├── loss/         # 损失函数
│   │   ├── embedding_lookup/ # 嵌入与查找
│   │   ├── attention/    # 注意力
│   │   ├── control_flow/ # 控制流
│   │   ├── tensor_manip/ # 张量操作
│   │   ├── data_gen/     # 数据生成
│   │   └── cast/         # 类型转换
│   ├── ir/               # IR 数据结构 (DAG/CFG)
│   ├── transform/        # 优化 Pass
│   ├── executor/         # 推理/训练执行器
│   ├── autograd/         # 自动微分引擎
│   └── frontend/         # 前端 (Hook/GGUF)
├── pybinds/              # Python 绑定
└── docs/                 # 开发文档
    ├── dev_ops.md        # 算子开发指南
    └── dev_ops_zh.md     # 算子开发指南 (中文)
```


## 开发指南

### 添加新算子

参考 `docs/dev_ops.md` 或 `docs/dev_ops_zh.md`，包含：

1. C++ 算子添加流程（如需）
2. Rust 算子开发模板
3. 反向传播实现
4. 量化支持
5. 测试用例

### 添加硬件平台支持

1. 实现 `Operator<T>` trait
2. 设备内存管理
3. 算子注册
4. 设备选择策略

详见 `docs/dev_hardware.md`（待补充）


## 支持平台

- macOS (Apple Silicon / Intel)
- Linux (x86_64 / ARM64)
- Windows (x86_64)

硬件加速支持：
- Apple GPU (Metal) - 计划中
- NVIDIA CUDA - 计划中
- AMD ROCm - 计划中
- 国产 NPU/GPU - 按需接入


## 许可证

Apache 2.0


## 贡献指南

1. Fork 仓库
2. 创建功能分支
3. 实现功能
4. 确保代码和注释中无英文外的其他语言
5. 确保代码**严格不超过80列**，包括注释
6. 保证每次commit，都是一个完整的，独立的，小型的改动
7. 确保commit msg无英文外的其他语言 
8. commit时加`-s`参数，以便生成Sign-off 
9. 提交 Pull Request

新算子开发请参考 `docs/dev_ops[_zh].md`。

硬件平台接入请参考 `docs/dev_platform[_zh].md`。

写代码之前请先阅读 `docs/rust_subset_guidelines[_zh].md`。

## 联系方式

- Issues: GitHub Issues
- 讨论: GitHub Discussions
