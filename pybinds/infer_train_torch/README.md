# InferTrain Torch

[![PyPI version](https://img.shields.io/pypi/v/infer_train_torch.svg)](https://pypi.org/project/infer_train_torch/)
[![Python version](https://img.shields.io/pypi/pyversions/infer_train_torch.svg)](https://pypi.org/project/infer_train_torch/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**InferTrain Torch** is a high-performance, lightweight deep learning framework designed for seamless integration with PyTorch. Built on a highly optimized underlying engine, it provides efficient functional operations, automatic differentiation (backward propagation), and advanced caching mechanisms to accelerate your training and inference workflows.

## Key Features

- High-Performance Engine: Powered by a custom-built backend engine for maximum computational efficiency.
- PyTorch Integration: Seamless interoperability with the PyTorch ecosystem.
- Comprehensive Functional Operations:
- Generic unary operations (e.g., abs, sqrt, neg).
- Generic binary operations (e.g., add, mul).
- Automatic Differentiation: Built-in backward propagation support for efficient gradient computation.
- Smart Caching: Automatically caches original functions to optimize execution and support advanced hooking mechanisms.

## Installation

You can install the latest release directly from PyPI:

```
pip install infer_train_torch
```

```
Platform Note: The current pre-built wheel (cp38-abi3-macosx_11_0_arm64) is compiled specifically for Apple Silicon (M-series) Macs running Python 3.8+. Support for other platforms (Intel Mac, Windows, Linux) will be added in future releases.
```

## Quick Start

```
import torch
import infer_train_torch

TODO: Add a brief example demonstrating a core feature,
such as a forward pass or a functional operation.
```

## License

This project is licensed under the Apache License 2.0.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
