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
import numpy as np
from typing import Any, Optional

from ._infer_train_torch import rust_engine

def to_engine_tensor(t: torch.Tensor) -> rust_engine.PyTensor:
    """PyTorch Tensor → PyTensor"""
    arr = t.detach().cpu().numpy()
    return rust_engine.PyTensor.from_numpy(arr)


def to_torch_tensor(t: rust_engine.PyTensor) -> torch.Tensor:
    """PyTensor → PyTorch Tensor"""
    arr = t.to_numpy()
    return torch.from_numpy(arr)


def wrap_tensor(t: torch.Tensor) -> torch.Tensor:
    """Attach the engine tensor to a PyTorch tensor."""
    if not hasattr(t, "_it_tensor"):
        t._it_tensor = to_engine_tensor(t)
    return t


def unwrap_tensor(t: torch.Tensor) -> Optional[rust_engine.PyTensor]:
    """Retrieve the engine tensor from a PyTorch tensor."""
    return getattr(t, "_it_tensor", None)


def is_engine_tensor(t: torch.Tensor) -> bool:
    """Check if the tensor is managed by the engine."""
    return hasattr(t, "_it_tensor") or t.device.type == "cpu"
