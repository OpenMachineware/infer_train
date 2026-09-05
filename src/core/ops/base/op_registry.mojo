# core/ops/base/op_registry.mojo
#
# The operator registry plus the dispatch functions that bridge the
# type-erased `AnyTensor` interface to the concrete per-dtype kernels.
#
# Each registered op name maps to a list of `OpInfo` (one per device).  The
# dispatchers convert `AnyTensor` -> `Tensor[dtype, rank]`, then call the CPU
# or GPU kernel selected by `device`; runtime dtype selection happens with a
# small `if` ladder over the supported floating dtypes.

from ...device import Device, get_default_device
from ...quantization import QuantFormat, QuantGranularity
from ...tensor import tensor_zeros
from ...utils import unimplemented
from .op_interface import (
    AnyTensor,
    OpInfo,
    from_any,
    to_any,
)
from ..cpu.matmul_cpu import (
    matmul_cpu_dynamic,
    matmul_weight_cpu,
    matmul_quantized_cpu as matmul_quantized_gguf_cpu,
)
from ..gpu.matmul_gpu import matmul_gpu_dynamic
from ..quantized.quant_types import QuantType
from ..cpu.rms_norm_cpu import rms_norm_cpu_dynamic
from ..gpu.rms_norm_gpu import rms_norm_gpu_dynamic
from ..cpu.softmax_cpu import softmax_cpu_dynamic
from ..gpu.softmax_gpu import softmax_gpu_dynamic
from ..cpu.embedding_cpu import embedding_cpu_dynamic
from ..gpu.embedding_gpu import embedding_gpu_dynamic
from ..cpu.rope_cpu import rope_cpu_dynamic
from ..gpu.rope_gpu import rope_gpu_dynamic
from ..cpu.add_cpu import add_cpu_dynamic, add_row_cpu
from ..gpu.add_gpu import add_gpu_dynamic
from ..cpu.swiglu_cpu import swiglu_cpu_dynamic
from ..gpu.swiglu_gpu import swiglu_gpu_dynamic
from ..gpu.fused_gpu import (
    fused_matmul_add_bias_gpu,
    fused_matmul_add_gpu,
    fused_matmul_rms_norm_gpu,
    fused_swiglu_matmul_gpu,
)
from ..attention.mha import mha_forward
from ..attention.kv_cache import KVCacheLayer
from ..quantized.matmul_quantized import (
    matmul_quantized_cpu,
    matmul_quantized_gpu,
)
from ..quantized.rms_norm_quantized import (
    rms_norm_quantized_cpu,
    rms_norm_quantized_gpu,
)
from ..fused.matmul_add import fused_matmul_add_bias, fused_matmul_add
from ..fused.matmul_rms_norm import fused_matmul_rms_norm
from ..fused.swiglu_matmul import fused_swiglu_matmul
from ..cpu.rms_norm_cpu import rms_norm_weight_cpu
from ..loss.cross_entropy import cross_entropy_forward
from .op_autograd import (
    matmul_fws_cpu,
    matmul_bwd_cpu,
    matmul_fws_gpu,
    matmul_bwd_gpu,
    lm_head_fws_cpu,
    lm_head_bwd_cpu,
    lm_head_fws_gpu,
    lm_head_bwd_gpu,
    add_fws_cpu,
    add_bwd_cpu,
    add_fws_gpu,
    add_bwd_gpu,
    add_bias_fws_cpu,
    add_bias_bwd_cpu,
    add_bias_fws_gpu,
    add_bias_bwd_gpu,
    rms_norm_fws_cpu,
    rms_norm_bwd_cpu,
    rms_norm_fws_gpu,
    rms_norm_bwd_gpu,
    rms_norm_weight_fws_cpu,
    rms_norm_weight_bwd_cpu,
    rms_norm_weight_fws_gpu,
    rms_norm_weight_bwd_gpu,
    softmax_fws_cpu,
    softmax_bwd_cpu,
    softmax_fws_gpu,
    softmax_bwd_gpu,
    rope_fws_cpu,
    rope_bwd_cpu,
    rope_fws_gpu,
    rope_bwd_gpu,
    swiglu_fws_cpu,
    swiglu_bwd_cpu,
    swiglu_fws_gpu,
    swiglu_bwd_gpu,
    embedding_fws_cpu,
    embedding_bwd_cpu,
    embedding_fws_gpu,
    embedding_bwd_gpu,
    swiglu_ffn_fws_cpu,
    swiglu_ffn_bwd_cpu,
    swiglu_ffn_fws_gpu,
    swiglu_ffn_bwd_gpu,
    mha_fws_cpu,
    mha_bwd_cpu,
    mha_fws_gpu,
    mha_bwd_gpu,
    mha_seq_fws_cpu,
    mha_seq_bwd_cpu,
    mha_seq_fws_gpu,
    mha_seq_bwd_gpu,
    cross_entropy_fws_cpu,
    cross_entropy_bwd_cpu,
    cross_entropy_fws_gpu,
    cross_entropy_bwd_gpu,
    identity_fws,
    identity_bwd,
    dynamic_quantize_fws_cpu,
    dynamic_quantize_bwd_cpu,
    dynamic_dequantize_fws_cpu,
    dynamic_dequantize_bwd_cpu,
)


# -- matmul dispatch --------------------------------------------------------


def _matmul_typed_cpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var a = from_any[dtype, 2](inputs[0])
    var b = from_any[dtype, 2](inputs[1])
    var out = matmul_cpu_dynamic[dtype](a, b)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def _matmul_typed_gpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var a = from_any[dtype, 2](inputs[0])
    var b = from_any[dtype, 2](inputs[1])
    var out = matmul_gpu_dynamic[dtype](a, b)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def matmul_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _matmul_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _matmul_typed_cpu[DType.float16](inputs)
    unimplemented("matmul_cpu: unsupported dtype")
    return List[AnyTensor]()


def matmul_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _matmul_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _matmul_typed_gpu[DType.float16](inputs)
    unimplemented("matmul_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- rms_norm dispatch ------------------------------------------------------


def _rms_norm_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var out = rms_norm_cpu_dynamic[dtype](x)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def _rms_norm_typed_gpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var out = rms_norm_gpu_dynamic[dtype](x)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def rms_norm_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _rms_norm_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _rms_norm_typed_cpu[DType.float16](inputs)
    unimplemented("rms_norm_cpu: unsupported dtype")
    return List[AnyTensor]()


def rms_norm_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _rms_norm_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _rms_norm_typed_gpu[DType.float16](inputs)
    unimplemented("rms_norm_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- softmax dispatch ------------------------------------------------------


def _softmax_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor] where dtype.is_floating_point():
    var x = from_any[dtype, 2](inputs[0])
    var out = softmax_cpu_dynamic[dtype](x)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def _softmax_typed_gpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor] where dtype.is_floating_point():
    var x = from_any[dtype, 2](inputs[0])
    var out = softmax_gpu_dynamic[dtype](x)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def softmax_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _softmax_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _softmax_typed_cpu[DType.float16](inputs)
    unimplemented("softmax_cpu: unsupported dtype")
    return List[AnyTensor]()


def softmax_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _softmax_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _softmax_typed_gpu[DType.float16](inputs)
    unimplemented("softmax_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- quantized matmul dispatch ---------------------------------------------


def _matmul_q_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    comptime format = QuantFormat.Q8_0
    comptime granularity = QuantGranularity.PerTensor
    comptime group_size = 0
    comptime is_symmetric = True
    var a = from_any[dtype, 2](inputs[0])
    var b_quant = from_any[dtype, 2](inputs[1])
    var scale = from_any[dtype, 1](inputs[2])
    var out = matmul_quantized_cpu[
        dtype, dtype, format, granularity, group_size, is_symmetric
    ](a, b_quant, scale)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def _matmul_q_typed_gpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    comptime format = QuantFormat.Q8_0
    comptime granularity = QuantGranularity.PerTensor
    comptime group_size = 0
    comptime is_symmetric = True
    var a = from_any[dtype, 2](inputs[0])
    var b_quant = from_any[dtype, 2](inputs[1])
    var scale = from_any[dtype, 1](inputs[2])
    var out = matmul_quantized_gpu[
        dtype, dtype, format, granularity, group_size, is_symmetric
    ](a, b_quant, scale)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def matmul_quantized_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _matmul_q_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _matmul_q_typed_cpu[DType.float16](inputs)
    unimplemented("matmul_quantized_cpu: unsupported dtype")
    return List[AnyTensor]()


def matmul_quantized_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _matmul_q_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _matmul_q_typed_gpu[DType.float16](inputs)
    unimplemented("matmul_quantized_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- matmul_quantized_cpu dispatch (M7: GGUF block formats) -----------------
#
# The erased interface cannot carry the comptime `quant_type`, so each
# registered entry is specialized at compile time: the dispatch function
# below fixes `quant_type` = Q4_K_M (the GGUF workhorse format) and
# `group_size` = 32 as comptime parameters.  Other formats are available
# through the typed API `matmul_quantized_gguf_cpu[dtype, QuantType.X, gs]`
# in `ops/cpu/matmul_cpu.mojo`.
#
# inputs: [a (dtype, 2), b_quant (uint8, 2), scale (dtype, 1)]


def _matmul_qcpu_typed[
    dtype: DType,
    quant_type: QuantType,
    group_size: Int,
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var a = from_any[dtype, 2](inputs[0])
    var b_quant = from_any[DType.uint8, 2](inputs[1])
    var scale = from_any[dtype, 1](inputs[2])
    var out = matmul_quantized_gguf_cpu[dtype, quant_type, group_size](
        a, b_quant, scale
    )
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def matmul_quantized_cpu_dispatch(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _matmul_qcpu_typed[DType.float32, QuantType.Q4_K_M, 32](inputs)
    if dtype == DType.float16:
        return _matmul_qcpu_typed[DType.float16, QuantType.Q4_K_M, 32](inputs)
    unimplemented("matmul_quantized_cpu: unsupported dtype")
    return List[AnyTensor]()


# -- quantized rms_norm dispatch -------------------------------------------


def _rms_norm_q_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    comptime format = QuantFormat.Q8_0
    comptime granularity = QuantGranularity.PerTensor
    comptime group_size = 0
    comptime is_symmetric = True
    var x_quant = from_any[dtype, 2](inputs[0])
    var scale = from_any[dtype, 1](inputs[1])
    var out = rms_norm_quantized_cpu[
        dtype, 0, format, granularity, group_size, is_symmetric
    ](x_quant, scale)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def rms_norm_quantized_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _rms_norm_q_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _rms_norm_q_typed_cpu[DType.float16](inputs)
    unimplemented("rms_norm_quantized_cpu: unsupported dtype")
    return List[AnyTensor]()


def _rms_norm_q_typed_gpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    comptime format = QuantFormat.Q8_0
    comptime granularity = QuantGranularity.PerTensor
    comptime group_size = 0
    comptime is_symmetric = True
    var x_quant = from_any[dtype, 2](inputs[0])
    var scale = from_any[dtype, 1](inputs[1])
    var out = rms_norm_quantized_gpu[
        dtype, 0, format, granularity, group_size, is_symmetric
    ](x_quant, scale)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def rms_norm_quantized_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _rms_norm_q_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _rms_norm_q_typed_gpu[DType.float16](inputs)
    unimplemented("rms_norm_quantized_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- embedding dispatch -----------------------------------------------------


def _embedding_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var tokens = from_any[DType.int32, 1](inputs[0])
    var table = from_any[dtype, 2](inputs[1])
    var out = embedding_cpu_dynamic[dtype](tokens, table)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def _embedding_typed_gpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var tokens = from_any[DType.int32, 1](inputs[0])
    var table = from_any[dtype, 2](inputs[1])
    var out = embedding_gpu_dynamic[dtype](tokens, table)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def embedding_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[1].dtype
    if dtype == DType.float32:
        return _embedding_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _embedding_typed_cpu[DType.float16](inputs)
    unimplemented("embedding_cpu: unsupported dtype")
    return List[AnyTensor]()


def embedding_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[1].dtype
    if dtype == DType.float32:
        return _embedding_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _embedding_typed_gpu[DType.float16](inputs)
    unimplemented("embedding_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- rope dispatch ----------------------------------------------------------


def _rope_typed_cpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 3](inputs[0])
    var pos_tensor = from_any[DType.float32, 1](inputs[1])
    var start_pos = Int(Float32(pos_tensor.get(0)))
    var out = rope_cpu_dynamic[dtype](x, start_pos)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 3](out))
    return results^


def _rope_typed_gpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 3](inputs[0])
    var pos_tensor = from_any[DType.float32, 1](inputs[1])
    var start_pos = Int(Float32(pos_tensor.get(0)))
    var out = rope_gpu_dynamic[dtype](x, start_pos)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 3](out))
    return results^


def rope_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _rope_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _rope_typed_cpu[DType.float16](inputs)
    unimplemented("rope_cpu: unsupported dtype")
    return List[AnyTensor]()


def rope_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _rope_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _rope_typed_gpu[DType.float16](inputs)
    unimplemented("rope_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- add dispatch -----------------------------------------------------------


def _add_typed_cpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var a = from_any[dtype, 2](inputs[0])
    var b = from_any[dtype, 2](inputs[1])
    var out = add_cpu_dynamic[dtype](a, b)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def _add_typed_gpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var a = from_any[dtype, 2](inputs[0])
    var b = from_any[dtype, 2](inputs[1])
    var out = add_gpu_dynamic[dtype](a, b)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def add_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _add_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _add_typed_cpu[DType.float16](inputs)
    unimplemented("add_cpu: unsupported dtype")
    return List[AnyTensor]()


def add_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _add_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _add_typed_gpu[DType.float16](inputs)
    unimplemented("add_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- add_bias dispatch (M5: registered so the optimizer IR can use it) ------


def _add_bias_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var bias = from_any[dtype, 1](inputs[1])
    var out = add_row_cpu[dtype](x, bias)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def add_bias_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _add_bias_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _add_bias_typed_cpu[DType.float16](inputs)
    unimplemented("add_bias_cpu: unsupported dtype")
    return List[AnyTensor]()


def add_bias_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    return add_bias_dispatch_cpu(inputs)


# -- swiglu dispatch --------------------------------------------------------


def _swiglu_typed_cpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var gate = from_any[dtype, 2](inputs[0])
    var up = from_any[dtype, 2](inputs[1])
    var out = swiglu_cpu_dynamic[dtype](gate, up)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def _swiglu_typed_gpu[dtype: DType](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var gate = from_any[dtype, 2](inputs[0])
    var up = from_any[dtype, 2](inputs[1])
    var out = swiglu_gpu_dynamic[dtype](gate, up)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def swiglu_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _swiglu_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _swiglu_typed_cpu[DType.float16](inputs)
    unimplemented("swiglu_cpu: unsupported dtype")
    return List[AnyTensor]()


def swiglu_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _swiglu_typed_gpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _swiglu_typed_gpu[DType.float16](inputs)
    unimplemented("swiglu_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- mha dispatch -----------------------------------------------------------
#
# inputs: [x, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache, start_pos(f32)]
# The K/V cache tensors are mutated in place; outputs: [attention_out].


def _mha_typed_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    """fp16 MHA through the erased interface (the KV cache is fp16)."""
    var x = from_any[DType.float16, 2](inputs[0])
    var wq = from_any[DType.float16, 2](inputs[1])
    var wk = from_any[DType.float16, 2](inputs[2])
    var wv = from_any[DType.float16, 2](inputs[3])
    var wo = from_any[DType.float16, 2](inputs[4])
    var bq = from_any[DType.float16, 1](inputs[5])
    var bk = from_any[DType.float16, 1](inputs[6])
    var bv = from_any[DType.float16, 1](inputs[7])
    var kc_tensor = from_any[DType.float16, 3](inputs[8])
    var vc_tensor = from_any[DType.float16, 3](inputs[9])
    var pos_tensor = from_any[DType.float32, 1](inputs[10])
    var start_pos = Int(Float32(pos_tensor.get(0)))
    var head_dim = kc_tensor.shape()[2]
    var n_kv_heads = kc_tensor.shape()[0]
    var n_heads = wq.shape()[1] // head_dim
    var cache = KVCacheLayer(0, 0, 0)
    cache.k = kc_tensor
    cache.v = vc_tensor
    cache.max_len = kc_tensor.shape()[1]
    cache.filled = start_pos
    var out = mha_forward(
        x,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        cache,
        start_pos,
        n_heads,
        n_kv_heads,
        head_dim,
        Float32(10000.0),
    )
    var results = List[AnyTensor]()
    results.append(to_any[DType.float16, 2](out))
    return results^


def mha_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    # The KV cache is fp16 by design, so the erased MHA path is fp16-only
    # (the typed f32 kernels remain available for non-cache use).
    var dtype = inputs[0].dtype
    if dtype == DType.float16:
        return _mha_typed_cpu(inputs)
    unimplemented("mha_cpu: unsupported dtype (cache is fp16)")
    return List[AnyTensor]()


def mha_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    # GPU attention falls back to the CPU kernel until the Metal backend
    # lands (M5); the registry entry keeps the dispatch point in place.
    return mha_dispatch_cpu(inputs)


# -- lm_head dispatch (x @ W) ----------------------------------------------


def lm_head_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var w = from_any[DType.float32, 2](inputs[1])
        var results = List[AnyTensor]()
        results.append(
            to_any[DType.float32, 2](matmul_weight_cpu[DType.float32](x, w))
        )
        return results^
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var w = from_any[DType.float16, 2](inputs[1])
        var results = List[AnyTensor]()
        results.append(
            to_any[DType.float16, 2](matmul_weight_cpu[DType.float16](x, w))
        )
        return results^
    unimplemented("lm_head_cpu: unsupported dtype")
    return List[AnyTensor]()


def lm_head_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    return lm_head_dispatch_cpu(inputs)


# -- swiglu_ffn dispatch (x -> down(silu(x@gate) * (x@up))) ----------------


def _swiglu_ffn_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var gate_w = from_any[dtype, 2](inputs[1])
    var up_w = from_any[dtype, 2](inputs[2])
    var down_w = from_any[dtype, 2](inputs[3])
    var g = matmul_weight_cpu[dtype](x, gate_w)
    var u = matmul_weight_cpu[dtype](x, up_w)
    var h = swiglu_cpu_dynamic[dtype](g, u)
    var out = matmul_weight_cpu[dtype](h, down_w)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def swiglu_ffn_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _swiglu_ffn_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _swiglu_ffn_typed_cpu[DType.float16](inputs)
    unimplemented("swiglu_ffn_cpu: unsupported dtype")
    return List[AnyTensor]()


def swiglu_ffn_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    return swiglu_ffn_dispatch_cpu(inputs)


# -- fused dispatch (M5) ----------------------------------------------------
#
# The fused kernels fold two ops into one pass; on the GPU side they fall
# back to the CPU kernels until the Metal backend lands.


def _fused_matmul_add_bias_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var w = from_any[dtype, 2](inputs[1])
    var bias = from_any[dtype, 1](inputs[2])
    var out = fused_matmul_add_bias[dtype](x, w, bias)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def fused_matmul_add_bias_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _fused_matmul_add_bias_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _fused_matmul_add_bias_typed_cpu[DType.float16](inputs)
    unimplemented("fused_matmul_add_bias_cpu: unsupported dtype")
    return List[AnyTensor]()


def fused_matmul_add_bias_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var w = from_any[DType.float32, 2](inputs[1])
        var bias = from_any[DType.float32, 1](inputs[2])
        var out = fused_matmul_add_bias_gpu[DType.float32](x, w, bias)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float32, 2](out))
        return results^
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var w = from_any[DType.float16, 2](inputs[1])
        var bias = from_any[DType.float16, 1](inputs[2])
        var out = fused_matmul_add_bias_gpu[DType.float16](x, w, bias)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float16, 2](out))
        return results^
    unimplemented("fused_matmul_add_bias_gpu: unsupported dtype")
    return List[AnyTensor]()


def _fused_matmul_add_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var w = from_any[dtype, 2](inputs[1])
    var b = from_any[dtype, 2](inputs[2])
    var out = fused_matmul_add[dtype](x, w, b)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def fused_matmul_add_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _fused_matmul_add_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _fused_matmul_add_typed_cpu[DType.float16](inputs)
    unimplemented("fused_matmul_add_cpu: unsupported dtype")
    return List[AnyTensor]()


def fused_matmul_add_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var w = from_any[DType.float32, 2](inputs[1])
        var b = from_any[DType.float32, 2](inputs[2])
        var out = fused_matmul_add_gpu[DType.float32](x, w, b)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float32, 2](out))
        return results^
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var w = from_any[DType.float16, 2](inputs[1])
        var b = from_any[DType.float16, 2](inputs[2])
        var out = fused_matmul_add_gpu[DType.float16](x, w, b)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float16, 2](out))
        return results^
    unimplemented("fused_matmul_add_gpu: unsupported dtype")
    return List[AnyTensor]()


def _fused_matmul_rms_norm_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var w = from_any[dtype, 2](inputs[1])
    var out = fused_matmul_rms_norm[dtype](x, w)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def fused_matmul_rms_norm_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _fused_matmul_rms_norm_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _fused_matmul_rms_norm_typed_cpu[DType.float16](inputs)
    unimplemented("fused_matmul_rms_norm_cpu: unsupported dtype")
    return List[AnyTensor]()


def fused_matmul_rms_norm_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var w = from_any[DType.float32, 2](inputs[1])
        var out = fused_matmul_rms_norm_gpu[DType.float32](x, w)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float32, 2](out))
        return results^
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var w = from_any[DType.float16, 2](inputs[1])
        var out = fused_matmul_rms_norm_gpu[DType.float16](x, w)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float16, 2](out))
        return results^
    unimplemented("fused_matmul_rms_norm_gpu: unsupported dtype")
    return List[AnyTensor]()


def _fused_swiglu_matmul_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var gate = from_any[dtype, 2](inputs[0])
    var up = from_any[dtype, 2](inputs[1])
    var w = from_any[dtype, 2](inputs[2])
    var out = fused_swiglu_matmul[dtype](gate, up, w)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def fused_swiglu_matmul_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _fused_swiglu_matmul_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _fused_swiglu_matmul_typed_cpu[DType.float16](inputs)
    unimplemented("fused_swiglu_matmul_cpu: unsupported dtype")
    return List[AnyTensor]()


def fused_swiglu_matmul_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var gate = from_any[DType.float32, 2](inputs[0])
        var up = from_any[DType.float32, 2](inputs[1])
        var w = from_any[DType.float32, 2](inputs[2])
        var out = fused_swiglu_matmul_gpu[DType.float32](gate, up, w)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float32, 2](out))
        return results^
    if dtype == DType.float16:
        var gate = from_any[DType.float16, 2](inputs[0])
        var up = from_any[DType.float16, 2](inputs[1])
        var w = from_any[DType.float16, 2](inputs[2])
        var out = fused_swiglu_matmul_gpu[DType.float16](gate, up, w)
        var results = List[AnyTensor]()
        results.append(to_any[DType.float16, 2](out))
        return results^
    unimplemented("fused_swiglu_matmul_gpu: unsupported dtype")
    return List[AnyTensor]()


# -- rms_norm_weight dispatch (M6) ------------------------------------------


def _rms_norm_weight_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var x = from_any[dtype, 2](inputs[0])
    var w = from_any[dtype, 1](inputs[1])
    var out = rms_norm_weight_cpu[dtype](x, w)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 2](out))
    return results^


def rms_norm_weight_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _rms_norm_weight_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _rms_norm_weight_typed_cpu[DType.float16](inputs)
    unimplemented("rms_norm_weight_cpu: unsupported dtype")
    return List[AnyTensor]()


def rms_norm_weight_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    return rms_norm_weight_dispatch_cpu(inputs)


# -- mha_seq dispatch (M6) ---------------------------------------------------


def mha_seq_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var fws = mha_seq_fws_cpu(inputs)
    return fws[0].copy()


def mha_seq_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    return mha_seq_dispatch_cpu(inputs)


# -- cross_entropy dispatch (M6) ---------------------------------------------


def _cross_entropy_typed_cpu[
    dtype: DType
](inputs: List[AnyTensor]) -> List[AnyTensor]:
    var logits = from_any[dtype, 2](inputs[0])
    var targets = from_any[DType.int32, 1](inputs[1])
    var loss = cross_entropy_forward[dtype](logits, targets)
    var results = List[AnyTensor]()
    results.append(to_any[dtype, 1](loss))
    return results^


def cross_entropy_dispatch_cpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        return _cross_entropy_typed_cpu[DType.float32](inputs)
    if dtype == DType.float16:
        return _cross_entropy_typed_cpu[DType.float16](inputs)
    unimplemented("cross_entropy_cpu: unsupported dtype")
    return List[AnyTensor]()


def cross_entropy_dispatch_gpu(inputs: List[AnyTensor]) -> List[AnyTensor]:
    return cross_entropy_dispatch_cpu(inputs)


# -- identity dispatch (M6) ---------------------------------------------------


def identity_dispatch(inputs: List[AnyTensor]) -> List[AnyTensor]:
    var results = List[AnyTensor]()
    for t in inputs:
        results.append(t)
    return results^


# -- dynamic quantize dispatch (M6 Phase 7) -----------------------------------


def dynamic_quantize_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var fws = dynamic_quantize_fws_cpu(inputs)
    return fws[0].copy()


def dynamic_quantize_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    return dynamic_quantize_dispatch_cpu(inputs)


def dynamic_dequantize_dispatch_cpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    var fws = dynamic_dequantize_fws_cpu(inputs)
    return fws[0].copy()


def dynamic_dequantize_dispatch_gpu(
    inputs: List[AnyTensor],
) -> List[AnyTensor]:
    return dynamic_dequantize_dispatch_cpu(inputs)


# -- placeholder stubs for not-found ops -----------------------------------


def _stub_forward(inputs: List[AnyTensor]) -> List[AnyTensor]:
    _ = inputs
    unimplemented("stub forward")
    return List[AnyTensor]()


def _stub_fws(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    _ = inputs
    unimplemented("stub forward_with_saved")
    return (List[AnyTensor](), List[AnyTensor]())


def _stub_backward(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    _ = grad_outputs
    _ = saved
    unimplemented("stub backward")
    return List[AnyTensor]()


# -- registry ---------------------------------------------------------------


struct OpRegistry(Movable):
    var _ops: Dict[String, List[OpInfo]]

    def __init__(out self):
        self._ops = Dict[String, List[OpInfo]]()

    def register(mut self, op_name: String, op_info: OpInfo):
        """Append an `OpInfo` for `op_name` (one entry per device)."""
        if op_name in self._ops:
            try:
                self._ops[op_name].append(op_info)
            except:
                pass
        else:
            var entries = List[OpInfo]()
            entries.append(op_info)
            self._ops[op_name] = entries^

    def get(
        self, op_name: String, preferred_device: Optional[Device] = None
    ) -> OpInfo:
        """Return the best matching `OpInfo`.

        * `preferred_device` wins when present.
        * Otherwise the default device's implementation is preferred.
        * Falls back to the first registered entry when neither matches.
        """
        var candidates = self._ops.get(op_name, List[OpInfo]())

        if preferred_device:
            var preferred = preferred_device.value()
            for op in candidates:
                if op.device == preferred:
                    return op

        var default_device = get_default_device()
        for op in candidates:
            if op.device == default_device:
                return op

        if len(candidates) > 0:
            return candidates[0]

        unimplemented("operator not registered: " + op_name)
        return OpInfo(
            op_name,
            _stub_forward,
            _stub_fws,
            _stub_backward,
            Device.CPU,
            0,
        )

    def register_default_ops(mut self):
        """Register the M1 operator set (CPU + GPU for each op)."""
        # matmul
        self.register(
            "matmul",
            OpInfo(
                "matmul",
                matmul_dispatch_cpu,
                matmul_fws_cpu,
                matmul_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "matmul",
            OpInfo(
                "matmul",
                matmul_dispatch_gpu,
                matmul_fws_gpu,
                matmul_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # rms_norm
        self.register(
            "rms_norm",
            OpInfo(
                "rms_norm",
                rms_norm_dispatch_cpu,
                rms_norm_fws_cpu,
                rms_norm_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "rms_norm",
            OpInfo(
                "rms_norm",
                rms_norm_dispatch_gpu,
                rms_norm_fws_gpu,
                rms_norm_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # matmul_quantized
        self.register(
            "matmul_quantized",
            OpInfo(
                "matmul_quantized",
                matmul_quantized_dispatch_cpu,
                _stub_fws,
                _stub_backward,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "matmul_quantized",
            OpInfo(
                "matmul_quantized",
                matmul_quantized_dispatch_gpu,
                _stub_fws,
                _stub_backward,
                Device.MetalGPU,
                10,
            ),
        )
        # matmul_quantized_cpu (M7: GGUF block-format quantized matmul;
        # quant_type/group_size are comptime-fixed per registration - see
        # the dispatch section header)
        self.register(
            "matmul_quantized_cpu",
            OpInfo(
                "matmul_quantized_cpu",
                matmul_quantized_cpu_dispatch,
                _stub_fws,
                _stub_backward,
                Device.CPU,
                0,
            ),
        )
        # rms_norm_quantized
        self.register(
            "rms_norm_quantized",
            OpInfo(
                "rms_norm_quantized",
                rms_norm_quantized_dispatch_cpu,
                _stub_fws,
                _stub_backward,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "rms_norm_quantized",
            OpInfo(
                "rms_norm_quantized",
                rms_norm_quantized_dispatch_gpu,
                _stub_fws,
                _stub_backward,
                Device.MetalGPU,
                10,
            ),
        )
        # embedding
        self.register(
            "embedding",
            OpInfo(
                "embedding",
                embedding_dispatch_cpu,
                embedding_fws_cpu,
                embedding_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "embedding",
            OpInfo(
                "embedding",
                embedding_dispatch_gpu,
                embedding_fws_gpu,
                embedding_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # rope
        self.register(
            "rope",
            OpInfo(
                "rope",
                rope_dispatch_cpu,
                rope_fws_cpu,
                rope_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "rope",
            OpInfo(
                "rope",
                rope_dispatch_gpu,
                rope_fws_gpu,
                rope_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # add
        self.register(
            "add",
            OpInfo(
                "add",
                add_dispatch_cpu,
                add_fws_cpu,
                add_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "add",
            OpInfo(
                "add",
                add_dispatch_gpu,
                add_fws_gpu,
                add_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # softmax (M5: registered so the optimizer IR can use it)
        self.register(
            "softmax",
            OpInfo(
                "softmax",
                softmax_dispatch_cpu,
                softmax_fws_cpu,
                softmax_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "softmax",
            OpInfo(
                "softmax",
                softmax_dispatch_gpu,
                softmax_fws_gpu,
                softmax_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # add_bias (M5)
        self.register(
            "add_bias",
            OpInfo(
                "add_bias",
                add_bias_dispatch_cpu,
                add_bias_fws_cpu,
                add_bias_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "add_bias",
            OpInfo(
                "add_bias",
                add_bias_dispatch_gpu,
                add_bias_fws_gpu,
                add_bias_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # swiglu
        self.register(
            "swiglu",
            OpInfo(
                "swiglu",
                swiglu_dispatch_cpu,
                swiglu_fws_cpu,
                swiglu_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "swiglu",
            OpInfo(
                "swiglu",
                swiglu_dispatch_gpu,
                swiglu_fws_gpu,
                swiglu_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # mha
        self.register(
            "mha",
            OpInfo(
                "mha",
                mha_dispatch_cpu,
                mha_fws_cpu,
                mha_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "mha",
            OpInfo(
                "mha",
                mha_dispatch_gpu,
                mha_fws_gpu,
                mha_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # lm_head
        self.register(
            "lm_head",
            OpInfo(
                "lm_head",
                lm_head_dispatch_cpu,
                lm_head_fws_cpu,
                lm_head_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "lm_head",
            OpInfo(
                "lm_head",
                lm_head_dispatch_gpu,
                lm_head_fws_gpu,
                lm_head_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # swiglu_ffn
        self.register(
            "swiglu_ffn",
            OpInfo(
                "swiglu_ffn",
                swiglu_ffn_dispatch_cpu,
                swiglu_ffn_fws_cpu,
                swiglu_ffn_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "swiglu_ffn",
            OpInfo(
                "swiglu_ffn",
                swiglu_ffn_dispatch_gpu,
                swiglu_ffn_fws_gpu,
                swiglu_ffn_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # fused_matmul_add_bias (M5)
        self.register(
            "fused_matmul_add_bias",
            OpInfo(
                "fused_matmul_add_bias",
                fused_matmul_add_bias_dispatch_cpu,
                _stub_fws,
                _stub_backward,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "fused_matmul_add_bias",
            OpInfo(
                "fused_matmul_add_bias",
                fused_matmul_add_bias_dispatch_gpu,
                _stub_fws,
                _stub_backward,
                Device.MetalGPU,
                10,
            ),
        )
        # fused_matmul_add (M5)
        self.register(
            "fused_matmul_add",
            OpInfo(
                "fused_matmul_add",
                fused_matmul_add_dispatch_cpu,
                _stub_fws,
                _stub_backward,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "fused_matmul_add",
            OpInfo(
                "fused_matmul_add",
                fused_matmul_add_dispatch_gpu,
                _stub_fws,
                _stub_backward,
                Device.MetalGPU,
                10,
            ),
        )
        # fused_matmul_rms_norm (M5)
        self.register(
            "fused_matmul_rms_norm",
            OpInfo(
                "fused_matmul_rms_norm",
                fused_matmul_rms_norm_dispatch_cpu,
                _stub_fws,
                _stub_backward,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "fused_matmul_rms_norm",
            OpInfo(
                "fused_matmul_rms_norm",
                fused_matmul_rms_norm_dispatch_gpu,
                _stub_fws,
                _stub_backward,
                Device.MetalGPU,
                10,
            ),
        )
        # fused_swiglu_matmul (M5)
        self.register(
            "fused_swiglu_matmul",
            OpInfo(
                "fused_swiglu_matmul",
                fused_swiglu_matmul_dispatch_cpu,
                _stub_fws,
                _stub_backward,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "fused_swiglu_matmul",
            OpInfo(
                "fused_swiglu_matmul",
                fused_swiglu_matmul_dispatch_gpu,
                _stub_fws,
                _stub_backward,
                Device.MetalGPU,
                10,
            ),
        )
        # rms_norm_weight (M6)
        self.register(
            "rms_norm_weight",
            OpInfo(
                "rms_norm_weight",
                rms_norm_weight_dispatch_cpu,
                rms_norm_weight_fws_cpu,
                rms_norm_weight_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "rms_norm_weight",
            OpInfo(
                "rms_norm_weight",
                rms_norm_weight_dispatch_gpu,
                rms_norm_weight_fws_gpu,
                rms_norm_weight_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # mha_seq (M6)
        self.register(
            "mha_seq",
            OpInfo(
                "mha_seq",
                mha_seq_dispatch_cpu,
                mha_seq_fws_cpu,
                mha_seq_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "mha_seq",
            OpInfo(
                "mha_seq",
                mha_seq_dispatch_gpu,
                mha_seq_fws_gpu,
                mha_seq_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # cross_entropy (M6)
        self.register(
            "cross_entropy",
            OpInfo(
                "cross_entropy",
                cross_entropy_dispatch_cpu,
                cross_entropy_fws_cpu,
                cross_entropy_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "cross_entropy",
            OpInfo(
                "cross_entropy",
                cross_entropy_dispatch_gpu,
                cross_entropy_fws_gpu,
                cross_entropy_bwd_gpu,
                Device.MetalGPU,
                10,
            ),
        )
        # identity (M6)
        self.register(
            "identity",
            OpInfo(
                "identity",
                identity_dispatch,
                identity_fws,
                identity_bwd,
                Device.CPU,
                0,
            ),
        )
        # dynamic_quantize (M6 Phase 7)
        self.register(
            "dynamic_quantize",
            OpInfo(
                "dynamic_quantize",
                dynamic_quantize_dispatch_cpu,
                dynamic_quantize_fws_cpu,
                dynamic_quantize_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "dynamic_quantize",
            OpInfo(
                "dynamic_quantize",
                dynamic_quantize_dispatch_gpu,
                dynamic_quantize_fws_cpu,
                dynamic_quantize_bwd_cpu,
                Device.MetalGPU,
                10,
            ),
        )
        # dynamic_dequantize (M6 Phase 7)
        self.register(
            "dynamic_dequantize",
            OpInfo(
                "dynamic_dequantize",
                dynamic_dequantize_dispatch_cpu,
                dynamic_dequantize_fws_cpu,
                dynamic_dequantize_bwd_cpu,
                Device.CPU,
                0,
            ),
        )
        self.register(
            "dynamic_dequantize",
            OpInfo(
                "dynamic_dequantize",
                dynamic_dequantize_dispatch_gpu,
                dynamic_dequantize_fws_cpu,
                dynamic_dequantize_bwd_cpu,
                Device.MetalGPU,
                10,
            ),
        )
