# InferTrain - Unified Inference and Training Engine

test

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![Chinese Docs](https://img.shields.io/badge/Chinese_Docs-Click_here-blue?style=for-the-badge)](./README_zh.md)

## Project Overview

InferTrain is an AI inference and training engine designed specifically for edge devices.

- **Model Privacy**: Inference runs locally on the device, keeping model weights and network structures private. Publishers can allow users to fine-tune without disclosing the network structure. Users can also save locally fine-tuned weights for continued use, achieving both privacy and personalization.
- **Unified Inference & Training**: The same codebase supports both inference and training, enabling weight updates during inference
- **Multi-Frontend Support**: PyTorch Hook/JIT Trace, GGUF
- **Multi-Precision Support**: F32/F64/F16/BF16/I8 quantization
- **Memory Reuse**: Byte-pool memory management, ideal for edge devices
- **ITM Private Format**: Proprietary binary model format with chunked storage, mmap, and incremental training
- **Pure Rust Implementation**: All operators implemented in Rust, no C++ dependencies

## Core Features

| Feature | Description |
|--|------------------------------------|
| Model Privacy | Publishers don't need to disclose network structure; users can keep personalized customizations private after local fine-tuning |
| Unified Inference & Training | Same model can infer and train; weights can be incrementally updated |
| Pure Rust Operators | 78+ operators, supporting F32/F64/F16/BF16/I8 |
| PyTorch Non-Invasive Integration | Just import, no user code modification required |
| GGUF Support | Import/export GGUF format, supports quantized model training |
| ITM Format | Proprietary binary format, chunked storage, mmap support |
| Memory Reuse | Byte-pool memory management, reduces training memory peak by 70%+ |
| Automatic Differentiation | Tape-based autograd engine |
| Parallel Execution | Rayon parallel, fully utilize multi-core - TODO |


## Quick Start

### Installation

```bash
# Install from source (requires Rust environment)
git clone https://github.com/yourname/infer_train.git
cd infer_train/pybinds/infer_train_torch
uv pip install -e .
```

### Python Usage

```python
import torch
import torch.nn as nn
import infer_train_torch as it  # ← Just add this line

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

# Inference (automatically uses engine)
output = model(x)

# Trace and export
dag = it.trace_model(model, [x])
it.export_model("model.itm", "my_model", trainable=True)

# Load and inference
model_file = it.PyModelFile.load("model.itm")
executor = it.PyExecutor.from_model_file(model_file)
result = executor.execute([x])
```

### GGUF Model Loading

```python
import infer_train_torch as it

# Import GGUF model
dag = it.import_gguf("llama-2-7b.gguf")

# Export to ITM (faster loading)
model_file = it.PyModelFile.new("llama", "gguf", dag)
model_file.export("llama.itm")

# Train GGUF model (infer while training)
trainer = it.Trainer.from_model_file(model_file)
loss = trainer.train_step(inputs, targets)
trainer.save("llama_updated.itm", trainable=True)

# Export back to GGUF
it.export_gguf("llama_updated.gguf", dag, "Q8_0")
```


## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                           │
│            (PyTorch Hook / GGUF Importer)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Rust Core Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    CFG IR   │→│    DAG IR   │→│   Executor/Trainer   │ │
│  │  (Control Flow Graph) │  │  (Data Flow Graph) │  │   (Inference/Training) │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              78+ Operators (Pure Rust)               │   │
│  │  math / linalg / activation / conv / normalization │   │
│  │  reduction / loss / embedding / attention / ...    │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Autograd (Tape + Backward)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ITM Storage Format                      │
│            (Chunked Storage / mmap / Incremental Training) │
└─────────────────────────────────────────────────────────────┘
```


## Directory Structure

```
infer_train/
├── src/
│   ├── ops/              # 78+ operator implementations
│   │   ├── math/         # Math operators
│   │   ├── linalg/       # Linear algebra
│   │   ├── activation/   # Activation functions
│   │   ├── conv_pool/    # Convolution and pooling
│   │   ├── normalization/# Normalization
│   │   ├── reduction/    # Reduction
│   │   ├── loss/         # Loss functions
│   │   ├── embedding_lookup/ # Embedding and lookup
│   │   ├── attention/    # Attention
│   │   ├── control_flow/ # Control flow
│   │   ├── tensor_manip/ # Tensor manipulation
│   │   ├── data_gen/     # Data generation
│   │   └── cast/         # Type conversion
│   ├── ir/               # IR data structures (DAG/CFG)
│   ├── transform/        # Optimization passes
│   ├── executor/         # Inference/training executor
│   ├── autograd/         # Autograd engine
│   └── frontend/         # Frontend (Hook/GGUF)
├── pybinds/              # Python bindings
└── docs/                 # Development documentation
    ├── dev_ops.md        # Operator development guide
    └── dev_ops_zh.md     # Operator development guide (Chinese)
```


## Development Guide

### Adding New Operators

Refer to [docs/dev_ops.md](./docs/dev_ops.md), which includes:

1. C++ operator addition workflow (if needed)
2. Rust operator development template
3. Backward propagation implementation
4. Quantization support
5. Test cases

### Adding Hardware Platform Support

1. Implement `Operator<T>` trait
2. Device memory management
3. Operator registration
4. Device selection strategy

See [docs/dev_platform.md](./docs/dev_platform.md) for details (to be added)


## Supported Platforms

- macOS (Apple Silicon / Intel)
- Linux (x86_64 / ARM64)
- Windows (x86_64)

Hardware acceleration support:
- Apple GPU (Metal) - Seeking volunteers
- NVIDIA CUDA - Seeking volunteers
- AMD ROCm - Seeking volunteers
- Other NPU/GPU/DSP/hardware accelerators - Seeking volunteers


## License

Apache 2.0


## Contributing Guide

1. Fork the repository
2. Create a feature branch
3. Implement the feature
4. Ensure no Chinese in code and comments
5. Ensure code **strictly does not exceed 80 columns**, including comments
6. Ensure each commit is a complete, independent, small change
7. Ensure commit messages have no Chinese
8. Add `-s` flag when committing to generate Sign-off
9. Submit Pull Request


Please refer to [TODO.md](TODO.md) for priority development tasks.

New operator development: refer to [docs/dev_ops.md](./docs/dev_ops.md).

Operator fusion optimization: refer to [docs/dev_fusion.md](./docs/dev_fusion.md).

Hardware platform integration: refer to [docs/dev_platform.md](./docs/dev_platform.md).

Please read [docs/rust_subset_guidelines.md](./docs/rust_subset_guidelines.md) before writing code.

## Contact

- Issues: GitHub Issues
- Discussion: GitHub Discussions
