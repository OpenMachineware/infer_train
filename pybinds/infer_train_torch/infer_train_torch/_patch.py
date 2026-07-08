# -*- coding: utf-8 -*-
# InferTrain - A Unified Inference and Training Engine
#
# Copyright (c) 2026 Jia Liu & InferTrain Contributors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of InferTrain.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, Callable, Dict
import functools

from ._infer_train_torch import rust_engine
from ._tensor import (to_engine_tensor, to_torch_tensor, wrap_tensor,
                      unwrap_tensor)


# ============================================================
# Cache the original function
# ============================================================

_ORIGINAL: Dict[str, Callable] = {}

def _save_original(name: str, func: Callable) -> None:
    _ORIGINAL[name] = func


def _get_original(name: str) -> Callable:
    return _ORIGINAL[name]


# ============================================================
# ops
# ============================================================

def _it_binary_op(self: Tensor, other: Any, op_name: str) -> Tensor:
    """Generic binary operations."""
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
    """Generic unary operations."""
    if self.device.type != "cpu":
        return _get_original(op_name)(self)

    a = to_engine_tensor(self)
    result = getattr(a, op_name)()
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_functional_op(input: Tensor, op_name: str, *args, **kwargs) -> Tensor:
    """Generic functional operations."""
    if input.device.type != "cpu":
        return _get_original(op_name)(input, *args, **kwargs)

    a = to_engine_tensor(input)
    result = getattr(a, op_name)(*args, **kwargs)
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


# ============================================================
# backward
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
    """Overwrite Tensor.backward"""
    if self.device.type == "cpu" and _IT_TRACER is not None:
        # Use engine backward propagation
        eng_tensor = unwrap_tensor(self)
        if eng_tensor is not None:
            _IT_TRACER.backward(eng_tensor)
            return

    # fallback to PyTorch
    _get_original("backward")(self, gradient, retain_graph)


# ============================================================
# optimizer
# ============================================================

def _it_optimizer_step(self) -> None:
    """Overwrite optimizer.step"""
    if _IT_TRACER is not None:
        _IT_TRACER.step()
    else:
        _get_original("optimizer_step")(self)


# ============================================================
# Apply patches
# ============================================================

def apply_patches(tracer=None):
    """Hack all ops"""
    global _IT_TRACER
    _IT_TRACER = tracer

    # Store original functional
    _save_original("add", Tensor.__add__)
    _save_original("sub", Tensor.__sub__)
    _save_original("mul", Tensor.__mul__)
    _save_original("matmul", Tensor.__matmul__)
    _save_original("backward", Tensor.backward)
    _save_original("optimizer_step", torch.optim.Optimizer.step)

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

    # Overwrite Tensor methods
    Tensor.__add__ = lambda self, other: _it_binary_op(self, other, "add")
    Tensor.__sub__ = lambda self, other: _it_binary_op(self, other, "sub")
    Tensor.__mul__ = lambda self, other: _it_binary_op(self, other, "mul")
    Tensor.__matmul__ = lambda self, other: _it_binary_op(self, other,
                                                          "matmul")
    Tensor.backward = _it_backward

    # Overwrite functional
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

    # Overwrite optimizer.step
    torch.optim.Optimizer.step = _it_optimizer_step


def _it_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
               groups=1):
    """Overwrite F.conv2d"""
    if input.device.type != "cpu":
        return _get_original("conv2d")(input, weight, bias, stride, padding,
                                       dilation, groups)

    a = to_engine_tensor(input)
    w = to_engine_tensor(weight)
    b = to_engine_tensor(bias) if bias is not None else None

    result = a.conv2d(w, b, stride, padding, dilation, groups)
    out = to_torch_tensor(result)
    wrap_tensor(out)
    return out


def _it_relu(input, inplace=False):
    """Overwrite F.relu"""
    if input.device.type != "cpu":
        return _get_original("relu")(input, inplace=inplace)

    # No inplace support, pass
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
# Remove ops patches
# ============================================================

def remove_patches():
    """Retore all ops"""
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
    # More ops...
