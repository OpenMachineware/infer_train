# Whole-tree import gate: builds every public module header.  Not part of
# `make test` (M3-M6 suites cover the behavior); run manually after big
# refactors.

from src.core.tensor import Tensor, tensor_zeros, tensor_copy
from src.core.device import Device, get_default_device
from src.core.memory import MemoryPool
from src.core.quantization import (
    QuantFormat, QuantGranularity, QuantizationInfo,
    quantize_static, quantize_dynamic, dequantize,
)
from src.core.sampler import Sampler, sample, greedy_sample
from src.core.tokenizer import Tokenizer
from src.core.graph import Graph, GraphNode, AttrValue
from src.core.ops.base.op_interface import AnyTensor, OpInfo, to_any, from_any
from src.core.ops.base.op_registry import OpRegistry
from src.core.ops.base.op_autograd import (
    no_grad_any, ones_like_any, accumulate_any, matmul_fws_cpu, matmul_bwd_cpu,
)
from src.core.ops.cpu.matmul_cpu import (
    matmul_cpu, matmul_cpu_dynamic, matmul_cpu_backward,
    matmul_weight_cpu_backward,
)
from src.core.ops.cpu.rms_norm_cpu import (
    rms_norm_cpu, rms_norm_cpu_dynamic, rms_norm_cpu_backward,
    rms_norm_weight_cpu, rms_norm_weight_cpu_backward,
)
from src.core.ops.cpu.softmax_cpu import (
    softmax_cpu, softmax_cpu_dynamic, softmax_cpu_backward,
)
from src.core.ops.cpu.rope_cpu import rope_cpu_dynamic, rope_cpu_backward
from src.core.ops.cpu.add_cpu import (
    add_cpu_dynamic, add_cpu_backward, add_row_cpu_backward,
)
from src.core.ops.cpu.swiglu_cpu import swiglu_cpu_dynamic, swiglu_cpu_backward
from src.core.ops.cpu.embedding_cpu import (
    embedding_cpu_dynamic, embedding_cpu_backward,
)
from src.core.ops.gpu.matmul_gpu import matmul_gpu
from src.core.ops.gpu.rms_norm_gpu import rms_norm_gpu
from src.core.ops.gpu.softmax_gpu import softmax_gpu
from src.core.ops.quantized.matmul_quantized import matmul_quantized_cpu
from src.core.ops.quantized.rms_norm_quantized import rms_norm_quantized_cpu
from src.core.ops.quantized.dynamic_quantize import (
    dynamic_quantize_symmetric, dynamic_quantize_asymmetric, dynamic_dequantize,
)
from src.core.ops.attention.mha import (
    multi_head_attention, mha_backward, _mha_seq_backward_typed,
)
from src.core.ops.attention.kv_cache import KVCache, kv_cache_append
from src.core.ops.loss.cross_entropy import (
    cross_entropy_loss, cross_entropy_forward, cross_entropy_backward,
)
from src.core.train_optimizer import AdamW, SGD, adamw_step_raw
from src.core.gradient_scaler import GradScaler
from src.core.training import TrainModel, TrainConfig, train_step, eval_step
from src.runtime.interpreter import Interpreter


def main():
    print("all modules import OK")
