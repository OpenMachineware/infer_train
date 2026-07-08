# python/infer_train_torch/__init__.py

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
from . import rust_engine

# 导出 Rust 类型
PyTensor = rust_engine.PyTensor
PyModelFile = rust_engine.PyModelFile
PyDagGraph = rust_engine.PyDagGraph
PyExecutor = rust_engine.PyExecutor
HookTracer = rust_engine.HookTracer

# 导入补丁
from ._patch import apply_patches, remove_patches, set_tracer
from ._tensor import to_engine_tensor, to_torch_tensor, wrap_tensor, unwrap_tensor

# ============================================================
# 全局 Tracer
# ============================================================

_tracer = None
_is_patched = False


def start_tracing(training_mode: bool = False):
    """开始追踪"""
    global _tracer, _is_patched

    _tracer = HookTracer()
    _tracer.set_training_mode(training_mode)
    _tracer.start_tracing()
    set_tracer(_tracer)

    if not _is_patched:
        apply_patches(_tracer)
        _is_patched = True


def stop_tracing():
    """停止追踪"""
    global _tracer
    if _tracer is not None:
        _tracer.stop_tracing()
        set_tracer(None)
        _tracer = None


def export_model(path: str, model_name: str = "model", trainable: bool = False):
    """导出追踪的模型"""
    if _tracer is not None:
        _tracer.trace_and_export(
            model=None,
            inputs=[],
            path=path,
            model_name=model_name,
            trainable=trainable,
        )


def get_dag():
    """获取追踪的 DAG"""
    if _tracer is not None:
        return _tracer.get_dag()
    return None


# ============================================================
# 自动启动 (用户只需 import)
# ============================================================

# ✅ 默认自动启动追踪 (推理模式)
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
    "to_engine_tensor",
    "to_torch_tensor",
    "wrap_tensor",
    "unwrap_tensor",
]

__version__ = "0.1.0"
