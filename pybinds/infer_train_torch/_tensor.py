# python/infer_train_torch/_tensor.py

import torch
import numpy as np
from typing import Any, Optional

# 导入 Rust 的 PyTensor
from . import rust_engine


def to_engine_tensor(t: torch.Tensor) -> rust_engine.PyTensor:
    """PyTorch Tensor → 引擎 PyTensor"""
    arr = t.detach().cpu().numpy()
    return rust_engine.PyTensor.from_numpy(arr)


def to_torch_tensor(t: rust_engine.PyTensor) -> torch.Tensor:
    """引擎 PyTensor → PyTorch Tensor"""
    arr = t.to_numpy()
    return torch.from_numpy(arr)


def wrap_tensor(t: torch.Tensor) -> torch.Tensor:
    """给 PyTorch Tensor 附加引擎 Tensor"""
    if not hasattr(t, "_it_tensor"):
        t._it_tensor = to_engine_tensor(t)
    return t


def unwrap_tensor(t: torch.Tensor) -> Optional[rust_engine.PyTensor]:
    """从 PyTorch Tensor 获取引擎 Tensor"""
    return getattr(t, "_it_tensor", None)


def is_engine_tensor(t: torch.Tensor) -> bool:
    """检查是否已被引擎接管"""
    return hasattr(t, "_it_tensor") or t.device.type == "cpu"
