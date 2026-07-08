"""
InferTrain Torch Plugin - 无感侵入 PyTorch 加速引擎

使用方法:
    import torch
    import infer_train_torch as it  # ← 只加这一行

    # 然后正常写 PyTorch 代码，自动走引擎
    model = MyModel()
    output = model(x)
    loss.backward()
    optimizer.step()
"""

# 导入 Rust 引擎
import torch
from ._infer_train_torch import rust_engine

# 导出 Rust 类型
PyTensor = rust_engine.PyTensor
PyModelFile = rust_engine.PyModelFile
PyDagGraph = rust_engine.PyDagGraph
PyExecutor = rust_engine.PyExecutor
HookTracer = rust_engine.HookTracer

# 导入补丁
from infer_train_torch._patch import apply_patches, remove_patches, set_tracer
from infer_train_torch._tensor import to_engine_tensor, to_torch_tensor, wrap_tensor, unwrap_tensor

# ============================================================
# 全局 Tracer
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
    """手动追踪模型，返回 DAG"""
    global _tracer, _traced_model, _traced_inputs

    print("[DEBUG] trace_model called (inside function)")
    print(f"[DEBUG] _tracer = {_tracer}")

    if _tracer is None:
        print("[DEBUG] Starting tracer...")
        start_tracing()

    print("[DEBUG] Saving model and inputs...")
    _traced_model = model
    _traced_inputs = inputs

    print("[DEBUG] Running forward pass...")
    with torch.no_grad():
        output = model(*inputs)
    print(f"[DEBUG] Forward pass done, output shape: {output.shape}")

    dag = _tracer.get_dag()
    print(f"[DEBUG] DAG: {dag}")
    return dag


def export_model(path: str, model_name: str = "model", trainable: bool = False):
    """导出追踪的模型"""
    global _tracer, _traced_model, _traced_inputs

    print("[DEBUG] export_model called")
    print(f"[DEBUG] _tracer = {_tracer}")
    print(f"[DEBUG] _traced_model = {_traced_model}")
    print(f"[DEBUG] _traced_inputs = {_traced_inputs}")

    if _tracer is None:
        raise RuntimeError("Tracer not initialized. Call start_tracing() first.")

    if _traced_model is None:
        raise RuntimeError("No model traced. Call trace_model() first.")

    print("[DEBUG] Exporting...")
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
    print("[DEBUG] Export done!")


def get_dag():
    """获取追踪的 DAG"""
    if _tracer is not None:
        return _tracer.get_dag()
    return None


# ============================================================
# 自动启动 (用户只需 import)
# ============================================================

# 默认自动启动追踪 (推理模式)
start_tracing(training_mode=False)

# 导出 API
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
