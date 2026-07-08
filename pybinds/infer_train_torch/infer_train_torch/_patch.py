# python/infer_train_torch/_patch.py

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, Callable, Dict
import functools

from ._infer_train_torch import rust_engine
from ._tensor import to_engine_tensor, to_torch_tensor, wrap_tensor, unwrap_tensor


# ============================================================
# 1. 缓存原始函数
# ============================================================

_ORIGINAL: Dict[str, Callable] = {}

def _save_original(name: str, func: Callable) -> None:
    _ORIGINAL[name] = func


def _get_original(name: str) -> Callable:
    return _ORIGINAL[name]


# ============================================================
# 2. 算子补丁
# ============================================================

def _it_binary_op(self: Tensor, other: Any, op_name: str) -> Tensor:
    """通用二元操作"""
    if self.device.type != "cpu":
        return _get_original(op_name)(self, other)

    a = to_engine_tensor(self)

    if isinstance(other, Tensor):
        b = to_engine_tensor(other)
    else:
        b = rust_engine.PyTensor([float(other)], [1])

    result = getattr(a, op_name)(b)
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_unary_op(self: Tensor, op_name: str) -> Tensor:
    """通用一元操作"""
    if self.device.type != "cpu":
        return _get_original(op_name)(self)

    a = to_engine_tensor(self)
    result = getattr(a, op_name)()
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_functional_op(input: Tensor, op_name: str, *args, **kwargs) -> Tensor:
    """通用 functional 操作"""
    if input.device.type != "cpu":
        return _get_original(op_name)(input, *args, **kwargs)

    a = to_engine_tensor(input)
    result = getattr(a, op_name)(*args, **kwargs)
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


# ============================================================
# 3. 反向传播补丁
# ============================================================

_IT_TRACER = None


def set_tracer(tracer):
    global _IT_TRACER
    _IT_TRACER = tracer


def _it_backward(
        self: Tensor,
        gradient: Optional[Tensor] = None,
        retain_graph: Optional[bool] = None,
) -> None:
    """覆盖 Tensor.backward"""
    if self.device.type == "cpu" and _IT_TRACER is not None:
        # 使用引擎的反向传播
        eng_tensor = unwrap_tensor(self)
        if eng_tensor is not None:
            _IT_TRACER.backward(eng_tensor)
            return

    # fallback 到 PyTorch
    _get_original("backward")(self, gradient, retain_graph)


# ============================================================
# 4. 优化器补丁
# ============================================================

def _it_optimizer_step(self) -> None:
    """覆盖 optimizer.step"""
    if _IT_TRACER is not None:
        _IT_TRACER.step()
    else:
        _get_original("optimizer_step")(self)


# ============================================================
# 5. 应用补丁
# ============================================================

def apply_patches(tracer=None):
    """应用所有补丁"""
    global _IT_TRACER
    _IT_TRACER = tracer

    # 保存原始函数
    _save_original("add", Tensor.__add__)
    _save_original("sub", Tensor.__sub__)
    _save_original("mul", Tensor.__mul__)
    _save_original("matmul", Tensor.__matmul__)
    _save_original("backward", Tensor.backward)
    _save_original("optimizer_step", torch.optim.Optimizer.step)

    # 保存 functional
    _save_original("relu", F.relu)
    _save_original("sigmoid", F.sigmoid)
    _save_original("tanh", F.tanh)
    _save_original("gelu", F.gelu)
    _save_original("silu", F.silu)
    _save_original("conv2d", F.conv2d)
    _save_original("linear", F.linear)
    _save_original("max_pool2d", F.max_pool2d)
    _save_original("avg_pool2d", F.avg_pool2d)
    _save_original("batch_norm", F.batch_norm)
    _save_original("layer_norm", F.layer_norm)
    _save_original("embedding", F.embedding)
    _save_original("dropout", F.dropout)
    _save_original("softmax", F.softmax)
    _save_original("log_softmax", F.log_softmax)

    # 覆盖 Tensor 方法
    Tensor.__add__ = lambda self, other: _it_binary_op(self, other, "add")
    Tensor.__sub__ = lambda self, other: _it_binary_op(self, other, "sub")
    Tensor.__mul__ = lambda self, other: _it_binary_op(self, other, "mul")
    Tensor.__matmul__ = lambda self, other: _it_binary_op(self, other, "matmul")
    Tensor.backward = _it_backward

    # 覆盖 functional
    # F.relu = lambda input: _it_functional_op(input, "relu")
    # F.sigmoid = lambda input: _it_functional_op(input, "sigmoid")
    # F.tanh = lambda input: _it_functional_op(input, "tanh")
    # F.gelu = lambda input: _it_functional_op(input, "gelu")
    # F.silu = lambda input: _it_functional_op(input, "silu")
    # F.conv2d = lambda input, weight, bias=None, **kwargs: _it_conv2d(input, weight, bias, **kwargs)
    # ... 其他 functional
    # 覆盖 functional（带完整参数）
    F.relu = _it_relu
    F.sigmoid = _it_sigmoid
    F.tanh = _it_tanh
    F.gelu = _it_gelu
    F.silu = _it_silu
    F.conv2d = _it_conv2d
    # F.linear = _it_linear
    # F.max_pool2d = _it_max_pool2d
    # F.avg_pool2d = _it_avg_pool2d
    # F.batch_norm = _it_batch_norm
    # F.layer_norm = _it_layer_norm
    # F.embedding = _it_embedding
    # F.dropout = _it_dropout
    # F.softmax = _it_softmax
    # F.log_softmax = _it_log_softmax

    # 覆盖 optimizer.step
    torch.optim.Optimizer.step = _it_optimizer_step


def _it_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """覆盖 F.conv2d"""
    if input.device.type != "cpu":
        return _get_original("conv2d")(input, weight, bias, stride, padding, dilation, groups)

    a = to_engine_tensor(input)
    w = to_engine_tensor(weight)
    b = to_engine_tensor(bias) if bias is not None else None

    result = a.conv2d(w, b, stride, padding, dilation, groups)
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_relu(input, inplace=False):
    """覆盖 F.relu"""
    if input.device.type != "cpu":
        return _get_original("relu")(input, inplace=inplace)

    # 引擎不支持 inplace，忽略
    a = to_engine_tensor(input)
    result = a.relu()
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out

def _it_sigmoid(input):
    if input.device.type != "cpu":
        return _get_original("sigmoid")(input)
    a = to_engine_tensor(input)
    result = a.sigmoid()
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_tanh(input):
    if input.device.type != "cpu":
        return _get_original("tanh")(input)
    a = to_engine_tensor(input)
    result = a.tanh()
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_gelu(input):
    if input.device.type != "cpu":
        return _get_original("gelu")(input)
    a = to_engine_tensor(input)
    result = a.gelu()
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_silu(input):
    if input.device.type != "cpu":
        return _get_original("silu")(input)
    a = to_engine_tensor(input)
    result = a.silu()
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


# ============================================================
# 6. 移除补丁
# ============================================================

def remove_patches():
    """移除所有补丁"""
    Tensor.__add__ = _get_original("add")
    Tensor.__sub__ = _get_original("sub")
    Tensor.__mul__ = _get_original("mul")
    Tensor.__matmul__ = _get_original("matmul")
    Tensor.backward = _get_original("backward")
    torch.optim.Optimizer.step = _get_original("optimizer_step")

    F.relu = _get_original("relu")
    F.sigmoid = _get_original("sigmoid")
    F.tanh = _get_original("tanh")
    F.gelu = _get_original("gelu")
    F.silu = _get_original("silu")
    F.conv2d = _get_original("conv2d")
    # ... 恢复所有
