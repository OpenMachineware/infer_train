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

from ._infer_train_torch import rust_engine

# Export Rust types
PyTensor = rust_engine.PyTensor
PyModelFile = rust_engine.PyModelFile
PyDagGraph = rust_engine.PyDagGraph
PyExecutor = rust_engine.PyExecutor
HookTracer = rust_engine.HookTracer

# Import patches
from infer_train_torch._patch import apply_patches, remove_patches, set_tracer
from infer_train_torch._tensor import (to_engine_tensor, to_torch_tensor,
                                       wrap_tensor, unwrap_tensor)

# ============================================================
# Global Tracer
# ============================================================

_tracer = None
_is_patched = False
_traced_model = None
_traced_inputs = None


def start_tracing(training_mode: bool = False):
    global _tracer, _is_patched
    _tracer = HookTracer()
    _tracer.set_training_mode(training_mode)
    _tracer.start_tracing()
    set_tracer(_tracer)
    if not _is_patched:
        apply_patches(_tracer)
        _is_patched = True


def stop_tracing():
    global _tracer
    if _tracer is not None:
        _tracer.stop_tracing()
        set_tracer(None)
        _tracer = None


def trace_model(model, inputs):
    """Manually trace a model and return the DAG."""
    global _tracer, _traced_model, _traced_inputs

    if _tracer is None:
        start_tracing()

    _traced_model = model
    _traced_inputs = inputs

    with torch.no_grad():
        output = model(*inputs)

    dag = _tracer.get_dag()
    return dag


def export_model(path: str, model_name: str = "model", trainable: bool = False):
    """Export the traced model."""
    global _tracer, _traced_model, _traced_inputs

    if _tracer is None:
        raise RuntimeError("Tracer not initialized. Call start_tracing() first.")

    if _traced_model is None:
        raise RuntimeError("No model traced. Call trace_model() first.")

    if trainable:
        _tracer.trace_and_export_trainable(
            _traced_model,
            _traced_inputs,
            path,
            model_name,
            "adam",
            0.001
        )
    else:
        _tracer.trace_and_export(
            _traced_model,
            _traced_inputs,
            path,
            model_name
        )


def get_dag():
    """Retrieve the traced DAG."""
    if _tracer is not None:
        return _tracer.get_dag()
    return None


# ============================================================
# Auto-start (Users only need to import this module)
# ============================================================

# Default to auto-start tracing in inference mode
start_tracing(training_mode=False)

# Export API
__all__ = [
    "PyTensor",
    "PyModelFile",
    "PyDagGraph",
    "PyExecutor",
    "HookTracer",
    "start_tracing",
    "stop_tracing",
    "export_model",
    "get_dag",
    "trace_model",
    "to_engine_tensor",
    "to_torch_tensor",
    "wrap_tensor",
    "unwrap_tensor",
]

__version__ = "0.1.0"
